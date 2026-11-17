"""Classify one image end to end under encryption and show every message that crosses the wire."""
import argparse
import time
from pathlib import Path

import numpy as np

from securedamage import CLASSES, data
from securedamage.he import Client, Server, PARAM_SETS
from securedamage.model import Model, im2col, preprocess

ROOT = Path(__file__).resolve().parents[1]
MB = 1024 * 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="A", choices=sorted(PARAM_SETS))
    ap.add_argument("--index", type=int, default=0, help="which test image to classify")
    ap.add_argument("--data-dir")
    args = ap.parse_args()

    model = Model.load(ROOT / "models" / "poly.json")
    _, (Xte, yte) = data.load(args.data_dir)
    image, label = Xte[args.index], yte[args.index]

    print(f"[client] parameter set {args.set}: {PARAM_SETS[args.set].label}")
    client = Client(PARAM_SETS[args.set])
    bundle = client.public_bundle()
    print(f"[client] -> server  setup bundle: " + ", ".join(f"{k} {len(v) / MB:.1f} MB" for k, v in bundle.items()))

    server = Server(bundle, model)
    print("[server] rebuilt context and evaluation keys from the bundle (no secret key)")

    start = time.perf_counter()
    query = client.encrypt_image(image)
    print(f"[client] -> server  query: {len(query)} ciphertexts, {sum(map(len, query)) / MB:.1f} MB "
          f"({time.perf_counter() - start:.2f} s to encrypt)")

    start = time.perf_counter()
    response = server.infer(query)
    print(f"[server] -> client  response: 1 ciphertext, {len(response) / MB:.2f} MB "
          f"({time.perf_counter() - start:.2f} s to evaluate, including one-time weight encoding)")

    logits = client.decrypt_logits(response)
    plain = model.forward(im2col(preprocess(image[None])))[0]
    print(f"\n[client] decrypted logits: {np.round(logits, 4)}")
    print(f"         plaintext logits: {np.round(plain, 4)}")
    print(f"         max |difference|: {np.max(np.abs(logits - plain)):.2e}")
    print(f"\nPrediction: {CLASSES[logits.argmax()]}   (true label: {CLASSES[label]})")


if __name__ == "__main__":
    main()
