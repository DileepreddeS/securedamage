"""The damage classifier, in plain numpy: training, plaintext inference, and export for HE.

Architecture (identical in plaintext and under encryption):
    input 32x32x3 in [-1, 1]
    conv  5x5, stride 2, 3 -> 5 channels   -> 5 x 14 x 14 = 980
    act   degree-2 polynomial (least-squares fit to ReLU on [-B, B])
    fc    980 -> 32
    act   degree-2 polynomial
    fc    32 -> 4 logits (none / minor / moderate / severe)
"""
import json
from pathlib import Path

import numpy as np

K, STRIDE, C_IN, C1, HIDDEN, N_CLASSES = 5, 2, 3, 5, 32, 4
OUT = (32 - K) // STRIDE + 1          # 14
POSITIONS = OUT * OUT                 # 196
PATCH = C_IN * K * K                  # 75
FLAT = C1 * POSITIONS                 # 980


def relu_poly_fit(bound=4.0):
    # Least-squares degree-2 fit of ReLU on [-bound, bound]: returns (c2, c1, c0).
    x = np.linspace(-bound, bound, 4001)
    c2, c1, c0 = np.polyfit(x, np.maximum(x, 0), 2)
    return float(c2), float(c1), float(c0)


def preprocess(images):
    return images * 2.0 - 1.0


def im2col(x):
    # x: (n, 32, 32, 3) -> (n, 196, 75); patch index j = ch * 25 + ky * 5 + kx, position p = oy * 14 + ox.
    n = x.shape[0]
    cols = np.empty((n, POSITIONS, PATCH))
    for oy in range(OUT):
        for ox in range(OUT):
            win = x[:, oy * STRIDE:oy * STRIDE + K, ox * STRIDE:ox * STRIDE + K, :]
            cols[:, oy * OUT + ox, :] = win.transpose(0, 3, 1, 2).reshape(n, PATCH)
    return cols


class Model:
    def __init__(self, activation="poly", seed=0, bound=4.0):
        rng = np.random.default_rng(seed)
        self.activation = activation
        self.coeffs = relu_poly_fit(bound)
        self.W1 = rng.normal(0, np.sqrt(2 / PATCH), (C1, PATCH))
        self.b1 = np.zeros(C1)
        self.W2 = rng.normal(0, np.sqrt(2 / FLAT), (HIDDEN, FLAT))
        self.b2 = np.zeros(HIDDEN)
        self.W3 = rng.normal(0, np.sqrt(2 / HIDDEN), (N_CLASSES, HIDDEN))
        self.b3 = np.zeros(N_CLASSES)

    def params(self):
        return ["W1", "b1", "W2", "b2", "W3", "b3"]

    def act(self, z):
        if self.activation == "relu":
            return np.maximum(z, 0)
        c2, c1, c0 = self.coeffs
        return c2 * z * z + c1 * z + c0

    def act_grad(self, z):
        if self.activation == "relu":
            return (z > 0).astype(z.dtype)
        c2, c1, _ = self.coeffs
        return 2 * c2 * z + c1

    def forward(self, cols, keep=False):
        z1 = cols @ self.W1.T + self.b1                      # (n, 196, 5)
        f1 = z1.transpose(0, 2, 1).reshape(len(cols), FLAT)  # channel-major: index c * 196 + p
        a1 = self.act(f1)
        z2 = a1 @ self.W2.T + self.b2
        a2 = self.act(z2)
        logits = a2 @ self.W3.T + self.b3
        if keep:
            return logits, (cols, f1, a1, z2, a2)
        return logits

    def predict(self, images):
        return self.forward(im2col(preprocess(images))).argmax(1)

    def accuracy(self, images, labels):
        return float((self.predict(images) == labels).mean())

    def train(self, X, y, epochs=40, batch=64, lr=2e-3, weight_decay=1e-3, seed=0, log=print):
        # Horizontal flips double the training set; damage looks the same mirrored.
        X, y = np.concatenate([X, X[:, :, ::-1, :]]), np.concatenate([y, y])
        cols_all = im2col(preprocess(X))
        rng = np.random.default_rng(seed)
        m = {k: np.zeros_like(getattr(self, k)) for k in self.params()}
        v = {k: np.zeros_like(getattr(self, k)) for k in self.params()}
        step = 0
        for epoch in range(epochs):
            order = rng.permutation(len(y))
            loss_sum = 0.0
            for i in range(0, len(y), batch):
                idx = order[i:i + batch]
                grads, loss = self._grads(cols_all[idx], y[idx])
                loss_sum += loss * len(idx)
                step += 1
                for k in self.params():
                    if k.startswith("W"):
                        grads[k] = grads[k] + weight_decay * getattr(self, k)
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    mh, vh = m[k] / (1 - 0.9 ** step), v[k] / (1 - 0.999 ** step)
                    setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + 1e-8))
            if log and (epoch % 5 == 4 or epoch == epochs - 1):
                log(f"  epoch {epoch + 1:>3}: loss {loss_sum / len(y):.4f}")

    def _grads(self, cols, y):
        n = len(y)
        logits, (cols, f1, a1, z2, a2) = self.forward(cols, keep=True)
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        loss = float(-np.log(p[np.arange(n), y] + 1e-12).mean())
        d = p.copy()
        d[np.arange(n), y] -= 1
        d /= n
        g = {"W3": d.T @ a2, "b3": d.sum(0)}
        d = (d @ self.W3) * self.act_grad(z2)
        g["W2"], g["b2"] = d.T @ a1, d.sum(0)
        d = (d @ self.W2) * self.act_grad(f1)
        d = d.reshape(n, C1, POSITIONS).transpose(0, 2, 1)   # back to (n, 196, 5)
        g["W1"] = np.einsum("npc,npj->cj", d, cols)
        g["b1"] = d.sum((0, 1))
        return g, loss

    def save(self, path):
        data = {k: getattr(self, k).tolist() for k in self.params()}
        data.update(activation=self.activation, coeffs=list(self.coeffs))
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        model = cls(activation=data["activation"])
        model.coeffs = tuple(data["coeffs"])
        for k in model.params():
            setattr(model, k, np.array(data[k]))
        return model
