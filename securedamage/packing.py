"""Slot layouts for encrypted inference, plus a numpy simulator of the exact HE algorithm.

One image uses 8192 CKKS slots, viewed as 8 blocks of 1024.

Conv layer (client-side im2col): the client sends 10 ciphertexts. In ciphertext g, block b holds
patch element j = 8g + b for every (channel c, position p) at slot c*196 + p. The server multiplies
by matching weight plaintexts, sums the 10 products, then rotate-and-sums across the 8 blocks, which
leaves the full convolution output (980 values) replicated in every block.

Dense layers: the "hybrid" diagonal method (Juvekar et al., GAZELLE) for an m x n matrix (m < n),
with baby-step/giant-step rotations, followed by a rotate-and-sum over n/m chunks.
"""
import numpy as np

from .model import C1, POSITIONS, PATCH, FLAT, HIDDEN, N_CLASSES

SLOTS = 8192
BLOCK = 1024
N_BLOCKS = SLOTS // BLOCK                    # 8
N_INPUT_CTS = -(-PATCH // N_BLOCKS)          # 10
FC1_BABY, FC1_GIANT = 8, HIDDEN // 8         # 8 x 4 = 32 diagonals


def rot(v, k):
    # CKKS EvalRotate(ct, k): slot i receives slot i + k (cyclic).
    return np.roll(v, -k)


def tile(v, period=None):
    period = period or len(v)
    return np.tile(v, SLOTS // period)


def he_folded_weights(model):
    # Fold the activation c2*z^2 + c1*z + c0 into the next layer, so the server only computes z^2 + k*z.
    c2, c1, c0 = model.coeffs
    if model.activation != "poly":
        raise ValueError("Only the polynomial-activation model can be evaluated under HE")
    return {
        "k": c1 / c2,
        "W1": model.W1, "b1": model.b1,
        "W2": c2 * model.W2, "b2": model.b2 + c0 * model.W2.sum(1),
        "W3": c2 * model.W3, "b3": model.b3 + c0 * model.W3.sum(1),
    }


def pack_input(cols):
    # cols: (196, 75) im2col patches of one image -> list of 10 slot vectors.
    vecs = []
    for g in range(N_INPUT_CTS):
        v = np.zeros(SLOTS)
        for b in range(N_BLOCKS):
            j = g * N_BLOCKS + b
            if j < PATCH:
                for c in range(C1):
                    start = b * BLOCK + c * POSITIONS
                    v[start:start + POSITIONS] = cols[:, j]
        vecs.append(v)
    return vecs


def conv_plaintexts(W1, b1):
    weights = []
    for g in range(N_INPUT_CTS):
        v = np.zeros(SLOTS)
        for b in range(N_BLOCKS):
            j = g * N_BLOCKS + b
            if j < PATCH:
                for c in range(C1):
                    start = b * BLOCK + c * POSITIONS
                    v[start:start + POSITIONS] = W1[c, j]
        weights.append(v)
    bias = np.zeros(BLOCK)
    for c in range(C1):
        bias[c * POSITIONS:(c + 1) * POSITIONS] = b1[c]
    return weights, tile(bias)


def hybrid_diagonals(W, n, baby):
    # W: (m, n_real) zero-padded to (m, n). Returns m diagonals, pre-rotated for baby-step/giant-step.
    m = W.shape[0]
    Wp = np.zeros((m, n))
    Wp[:, :W.shape[1]] = W
    j = np.arange(n)
    diags = []
    for k in range(m):
        d = Wp[j % m, (j + k) % n]
        giant = (k // baby) * baby
        diags.append(tile(np.roll(d, giant), n))
    return diags


def fc_plaintexts(fw):
    fc1 = hybrid_diagonals(fw["W2"], BLOCK, FC1_BABY)
    fc2 = hybrid_diagonals(fw["W3"], HIDDEN, N_CLASSES)
    return fc1, tile(fw["b2"], HIDDEN), fc2, tile(fw["b3"], N_CLASSES)


def chunk_sums(n, m):
    # Rotation steps for summing n/m chunks of size m: n/2, n/4, ..., m.
    steps, s = [], n // 2
    while s >= m:
        steps.append(s)
        s //= 2
    return steps


def rotation_indices():
    idx = {BLOCK * 2 ** i for i in range(3)}                                    # conv block sum
    idx |= set(range(1, FC1_BABY)) | {g * FC1_BABY for g in range(1, FC1_GIANT)}
    idx |= set(chunk_sums(BLOCK, HIDDEN))
    idx |= set(range(1, N_CLASSES)) | set(chunk_sums(HIDDEN, N_CLASSES))
    return sorted(idx)


def bsgs_matvec(x, diags, baby, rotate, mul, add):
    # Baby-step/giant-step diagonal matvec, written against abstract rotate/mul/add so numpy and HE share it.
    babies = [x] + [rotate(x, b) for b in range(1, baby)]
    out = None
    for g in range(len(diags) // baby):
        inner = None
        for b in range(baby):
            term = mul(babies[b], diags[g * baby + b])
            inner = term if inner is None else add(inner, term)
        if g:
            inner = rotate(inner, g * baby)
        out = inner if out is None else add(out, inner)
    return out


def simulate(cols, fw):
    # Runs the exact slot algorithm the server runs under HE, but on plain numpy vectors.
    k = fw["k"]
    conv_w, conv_b = conv_plaintexts(fw["W1"], fw["b1"])
    fc1, b2, fc2, b3 = fc_plaintexts(fw)
    x = sum(v * w for v, w in zip(pack_input(cols), conv_w))
    for s in (BLOCK, 2 * BLOCK, 4 * BLOCK):
        x = x + rot(x, s)
    x = x + conv_b
    x = x * x + k * x
    x = bsgs_matvec(x, fc1, FC1_BABY, rot, np.multiply, np.add)
    for s in chunk_sums(BLOCK, HIDDEN):
        x = x + rot(x, s)
    x = x + b2
    x = x * x + k * x
    x = bsgs_matvec(x, fc2, N_CLASSES, rot, np.multiply, np.add)
    for s in chunk_sums(HIDDEN, N_CLASSES):
        x = x + rot(x, s)
    x = x + b3
    return x[:N_CLASSES]
