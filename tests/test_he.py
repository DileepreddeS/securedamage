"""Encrypted inference (parameter set A) against the plaintext model."""
import numpy as np
import pytest

from securedamage import data
from securedamage.he import Client, Server, PARAM_SETS, reset_key_cache
from securedamage.model import Model, im2col, preprocess


@pytest.fixture(scope="module")
def setup():
    reset_key_cache()
    model = Model("poly", seed=7)
    (X, y), _ = data.synthetic_dataset(40, seed=11), None
    model.train(X, y, epochs=2, log=None)
    client = Client(PARAM_SETS["A"])
    server = Server(client.public_bundle(), model)
    return model, client, server, X


def test_encrypted_logits_match_plaintext(setup):
    model, client, server, X = setup
    ref = model.forward(im2col(preprocess(X[:3])))
    for i in range(3):
        logits = client.decrypt_logits(server.infer(client.encrypt_image(X[i])))
        assert np.max(np.abs(logits - ref[i])) < 1e-4
        assert logits.argmax() == ref[i].argmax()


def test_parameters_meet_128_bit_standard(setup):
    _, client, _, _ = setup
    info = client.info()
    assert info["N"] == 16384 and info["log2_Q"] < 438


def test_bundle_contains_no_secret_key(setup):
    _, client, _, _ = setup
    bundle = client.public_bundle()
    assert set(bundle) == {"context", "public_key", "relin_key", "rotation_keys"}


def test_other_key_cannot_decrypt(setup):
    model, client, server, X = setup
    response = server.infer(client.encrypt_image(X[0]))
    ref = model.forward(im2col(preprocess(X[:1])))[0]
    stranger = Client(PARAM_SETS["A"])
    try:
        wrong = stranger.decrypt_logits(response)
    except RuntimeError as e:
        assert "approximation error is too high" in str(e)
    else:
        assert np.max(np.abs(wrong - ref)) > 1.0
