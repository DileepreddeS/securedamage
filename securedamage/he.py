"""CKKS client and server (OpenFHE).

The client owns the secret key. It sends the crypto context, public key and evaluation keys once,
then one encrypted query (10 ciphertexts) per image, and decrypts the 4 returned logits.
The server holds the model weights in plaintext and never holds key material that can decrypt.
"""
import math
import time
from dataclasses import dataclass

import numpy as np
import openfhe as fhe

from . import packing as P
from .model import im2col, preprocess, N_CLASSES

DEPTH = 5


@dataclass(frozen=True)
class ParamSet:
    name: str
    label: str
    scale_bits: int
    first_mod_bits: int
    security: str
    digits: int = 0

    def build(self):
        params = fhe.CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(DEPTH)
        params.SetScalingModSize(self.scale_bits)
        params.SetFirstModSize(self.first_mod_bits)
        params.SetBatchSize(P.SLOTS)
        params.SetSecurityLevel(getattr(fhe, f"HEStd_{self.security}_classic"))
        if self.digits:
            params.SetNumLargeDigits(self.digits)
        cc = fhe.GenCryptoContext(params)
        for f in (fhe.PKESchemeFeature.PKE, fhe.PKESchemeFeature.KEYSWITCH, fhe.PKESchemeFeature.LEVELEDSHE):
            cc.Enable(f)
        return cc


PARAM_SETS = {
    "A": ParamSet("A", "128-bit, scale 2^40", 40, 60, "128", digits=4),
    "B": ParamSet("B", "128-bit, scale 2^50", 50, 60, "128"),
    "C": ParamSet("C", "192-bit, scale 2^40", 40, 60, "192"),
}


def to_bytes(obj):
    return fhe.Serialize(obj, fhe.BINARY)


def ciphertext_from_bytes(data):
    return fhe.DeserializeCiphertextString(data, fhe.BINARY)


def reset_key_cache():
    # OpenFHE keeps evaluation keys in a process-wide cache; clear it between parameter sets.
    fhe.ClearEvalMultKeys()
    fhe.CryptoContext.ClearEvalAutomorphismKeys()


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.start = time.perf_counter()

            def __exit__(self, *exc):
                timer.t[name] = timer.t.get(name, 0.0) + time.perf_counter() - self.start

        return _Ctx()


class Client:
    def __init__(self, pset):
        self.pset = pset
        self.timer = Timer()
        with self.timer("context"):
            self.cc = pset.build()
        with self.timer("keygen"):
            self.keys = self.cc.KeyGen()
        with self.timer("evalkeygen"):
            self.cc.EvalMultKeyGen(self.keys.secretKey)
            self.cc.EvalRotateKeyGen(self.keys.secretKey, P.rotation_indices())

    def info(self):
        return {"N": self.cc.GetRingDimension(), "log2_Q": math.log2(self.cc.GetModulus()),
                "slots": P.SLOTS, "depth": DEPTH, "scale_bits": self.pset.scale_bits,
                "security": self.pset.security}

    def public_bundle(self):
        # Everything the server receives at setup, as bytes. The secret key is never included.
        tag = self.keys.secretKey.GetKeyTag()
        return {"context": to_bytes(self.cc),
                "public_key": to_bytes(self.keys.publicKey),
                "relin_key": fhe.SerializeEvalMultKeyString(fhe.BINARY, tag),
                "rotation_keys": fhe.SerializeEvalAutomorphismKeyString(fhe.BINARY, tag)}

    def encrypt_image(self, image):
        # image: (32, 32, 3) in [0, 1]. Preprocessing and im2col happen on the client, in the clear.
        cols = im2col(preprocess(image[None]))[0]
        return [to_bytes(self.cc.Encrypt(self.keys.publicKey, self.cc.MakeCKKSPackedPlaintext(v.tolist())))
                for v in P.pack_input(cols)]

    def decrypt_logits(self, response):
        pt = self.cc.Decrypt(self.keys.secretKey, ciphertext_from_bytes(response))
        pt.SetLength(N_CLASSES)
        return np.array(pt.GetRealPackedValue())


class Server:
    def __init__(self, bundle, model):
        # Rebuilt from the client's public bundle only; weights are folded and packed once.
        reset_key_cache()
        self.cc = fhe.DeserializeCryptoContextString(bundle["context"], fhe.BINARY)
        fhe.DeserializeEvalMultKeyString(bundle["relin_key"], fhe.BINARY)
        fhe.DeserializeEvalAutomorphismKeyString(bundle["rotation_keys"], fhe.BINARY)
        fw = P.he_folded_weights(model)
        self.k = fw["k"]
        conv_w, conv_b = P.conv_plaintexts(fw["W1"], fw["b1"])
        fc1, b2, fc2, b3 = P.fc_plaintexts(fw)
        self.raw = {"conv": conv_w, "fc1": fc1, "fc2": fc2, "b_conv": [conv_b], "b_fc1": [b2], "b_fc2": [b3]}
        self.encoded = {}
        self.timer = Timer()

    def _weights(self, name, ct):
        # Plaintexts are encoded once, at the level and scale degree of the ciphertext they meet.
        key = (name, ct.GetLevel(), ct.GetNoiseScaleDeg())
        if key not in self.encoded:
            deg = ct.GetNoiseScaleDeg() if name.startswith("b_") else 1
            self.encoded[key] = [self.cc.MakeCKKSPackedPlaintext(list(v), deg, ct.GetLevel())
                                 for v in self.raw[name]]
        return self.encoded[key]

    def _add_bias(self, ct, name):
        return self.cc.EvalAdd(ct, self._weights("b_" + name, ct)[0])

    def _activation(self, ct):
        return self.cc.EvalAdd(self.cc.EvalSquare(ct), self.cc.EvalMult(ct, self.k))

    def _rotate_sum(self, ct, steps):
        for s in steps:
            ct = self.cc.EvalAdd(ct, self.cc.EvalRotate(ct, s))
        return ct

    def infer(self, query):
        cc, t = self.cc, self.timer
        with t("deserialize"):
            cts = [ciphertext_from_bytes(b) for b in query]
        with t("conv"):
            conv_w = self._weights("conv", cts[0])
            x = cc.EvalMult(cts[0], conv_w[0])
            for ct, w in zip(cts[1:], conv_w[1:]):
                x = cc.EvalAdd(x, cc.EvalMult(ct, w))
            x = self._rotate_sum(x, (4 * P.BLOCK, 2 * P.BLOCK, P.BLOCK))
            x = self._add_bias(x, "conv")
        with t("act1"):
            x = self._activation(x)
        with t("fc1"):
            x = P.bsgs_matvec(x, self._weights("fc1", x), P.FC1_BABY, cc.EvalRotate, cc.EvalMult, cc.EvalAdd)
            x = self._rotate_sum(x, P.chunk_sums(P.BLOCK, P.HIDDEN))
            x = self._add_bias(x, "fc1")
        with t("act2"):
            x = self._activation(x)
        with t("fc2"):
            x = P.bsgs_matvec(x, self._weights("fc2", x), N_CLASSES, cc.EvalRotate, cc.EvalMult, cc.EvalAdd)
            x = self._rotate_sum(x, P.chunk_sums(P.HIDDEN, N_CLASSES))
            x = self._add_bias(x, "fc2")
        with t("serialize"):
            return to_bytes(x)
