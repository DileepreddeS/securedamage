"""The slot-packing algorithm, run on plain numpy vectors, must equal the dense model exactly."""
import numpy as np

from securedamage import packing as P
from securedamage.model import Model, im2col


def random_model(seed):
    model = Model("poly", seed=seed)
    rng = np.random.default_rng(seed)
    for k in ("b1", "b2", "b3"):
        setattr(model, k, rng.normal(size=getattr(model, k).shape))
    return model


def test_simulated_slot_algorithm_equals_model():
    rng = np.random.default_rng(0)
    for seed in range(3):
        model = random_model(seed)
        fw = P.he_folded_weights(model)
        cols = im2col(rng.uniform(-1, 1, size=(4, 32, 32, 3)))
        ref = model.forward(cols)
        for i in range(4):
            assert np.allclose(P.simulate(cols[i], fw), ref[i], rtol=1e-10, atol=1e-10)


def test_activation_folding_is_exact():
    model = random_model(5)
    fw = P.he_folded_weights(model)
    z = np.linspace(-20, 20, 101)
    c2, c1, c0 = model.coeffs
    W = model.W3
    lhs = W @ np.vstack([model.act(z)] * W.shape[1])
    rhs = fw["W3"] @ np.vstack([z * z + fw["k"] * z] * W.shape[1]) + (fw["b3"] - model.b3)[:, None]
    assert np.allclose(lhs, rhs)


def test_input_fits_in_slots_and_rotation_keys_cover_algorithm():
    assert P.N_INPUT_CTS * P.N_BLOCKS >= 75
    assert 5 * 196 <= P.BLOCK
    used = set()
    record = lambda v, k: (used.add(k), P.rot(v, k))[1]
    fw = P.he_folded_weights(random_model(1))
    x = np.ones(P.SLOTS)
    fc1, _, fc2, _ = P.fc_plaintexts(fw)
    P.bsgs_matvec(x, fc1, P.FC1_BABY, record, np.multiply, np.add)
    P.bsgs_matvec(x, fc2, 4, record, np.multiply, np.add)
    used |= {P.BLOCK, 2 * P.BLOCK, 4 * P.BLOCK}
    used |= set(P.chunk_sums(P.BLOCK, 32)) | set(P.chunk_sums(32, 4))
    assert used <= set(P.rotation_indices())
