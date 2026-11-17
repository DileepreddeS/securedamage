"""Plaintext model: im2col layout and analytic gradients."""
import numpy as np

from securedamage.model import Model, im2col, K, STRIDE, C1, POSITIONS, PATCH, OUT


def test_im2col_matches_direct_convolution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 32, 32, 3))
    W = rng.normal(size=(C1, PATCH))
    cols = im2col(x)
    kernel = W.reshape(C1, 3, K, K)
    for n in range(2):
        for c in range(C1):
            for oy in range(0, OUT, 5):
                for ox in range(0, OUT, 3):
                    win = x[n, oy * STRIDE:oy * STRIDE + K, ox * STRIDE:ox * STRIDE + K, :]
                    direct = np.sum(win.transpose(2, 0, 1) * kernel[c])
                    assert np.isclose(cols[n, oy * OUT + ox] @ W[c], direct)
    assert cols.shape == (2, POSITIONS, PATCH)


def test_analytic_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    model = Model("poly", seed=3)
    for k in model.params():
        setattr(model, k, getattr(model, k) * 0.3)
    cols = im2col(rng.uniform(-1, 1, size=(3, 32, 32, 3)))
    y = np.array([0, 2, 3])
    grads, _ = model._grads(cols, y)
    eps = 1e-6
    for name in model.params():
        param = getattr(model, name)
        for idx in [tuple(rng.integers(0, s) for s in param.shape) for _ in range(4)]:
            old = param[idx]
            param[idx] = old + eps
            up = model._grads(cols, y)[1]
            param[idx] = old - eps
            down = model._grads(cols, y)[1]
            param[idx] = old
            assert np.isclose(grads[name][idx], (up - down) / (2 * eps), rtol=1e-4, atol=1e-7), name
