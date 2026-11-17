"""Datasets: a reproducible synthetic damage dataset, or real images from a folder."""
from pathlib import Path

import numpy as np

from . import CLASSES, IMAGE_SIZE


def _panel(rng, s):
    # A painted body panel: base colour, vertical shading, a specular highlight band, sensor noise.
    base = rng.uniform(0.15, 0.85, size=3)
    yy, xx = np.mgrid[0:s, 0:s] / (s - 1)
    shade = 1.0 + rng.uniform(-0.25, 0.25) * (yy - 0.5)
    angle = rng.uniform(-0.6, 0.6)
    band = np.exp(-((yy - 0.5 - angle * (xx - 0.5)) ** 2) / (2 * rng.uniform(0.02, 0.06) ** 2))
    img = base[None, None, :] * shade[..., None] + rng.uniform(0.05, 0.25) * band[..., None]
    return img + rng.normal(0, 0.03, size=img.shape)


def _scratch(img, rng):
    # A thin bright line at a random angle (paint scraped down to primer).
    s = img.shape[0]
    x0, y0 = rng.uniform(0, s, 2)
    theta = rng.uniform(0, np.pi)
    length = rng.uniform(6, 18)
    for t in np.linspace(0, length, int(length * 3)):
        x, y = int(x0 + t * np.cos(theta)), int(y0 + t * np.sin(theta))
        if 0 <= x < s and 0 <= y < s:
            img[y, x] = 0.6 * img[y, x] + 0.4 * np.array([0.92, 0.92, 0.88])


def _dent(img, rng, scale):
    # A dent: a dark elliptical shadow with a lighter rim on one side.
    s = img.shape[0]
    cx, cy = rng.uniform(6, s - 6, 2)
    rx, ry = rng.uniform(2.5, 5.5, 2) * scale
    yy, xx = np.mgrid[0:s, 0:s]
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    shadow = np.exp(-d * 1.5)
    rim = np.exp(-((((xx - cx + 1.5) / rx) ** 2 + ((yy - cy - 1.5) / ry) ** 2) - 1.0) ** 2 * 4)
    img *= (1 - 0.55 * shadow)[..., None]
    img += 0.12 * rim[..., None]


def _crack(img, rng):
    # A jagged dark crack (cracked bumper or panel).
    s = img.shape[0]
    x, y = rng.uniform(4, s - 4, 2)
    for _ in range(int(rng.integers(12, 24))):
        x += rng.uniform(-1.5, 1.5)
        y += rng.uniform(0.3, 1.5)
        xi, yi = int(x), int(y)
        if 0 <= xi < s and 0 <= yi < s:
            img[yi, xi] *= 0.25


def _rust(img, rng):
    # A rust / exposed-metal patch.
    s = img.shape[0]
    cx, cy = rng.uniform(4, s - 4, 2)
    r = rng.uniform(2, 4.5)
    yy, xx = np.mgrid[0:s, 0:s]
    mask = (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2) & (rng.random((s, s)) < 0.8)
    img[mask] = 0.5 * img[mask] + 0.5 * np.array([0.55, 0.27, 0.1])


def synthetic_image(label, rng, s=IMAGE_SIZE):
    img = _panel(rng, s)
    if label == 1:
        for _ in range(rng.integers(1, 3)):
            _scratch(img, rng)
        if rng.random() < 0.3:
            _dent(img, rng, 0.6)
    elif label == 2:
        for _ in range(rng.integers(1, 3)):
            _scratch(img, rng)
        for _ in range(rng.integers(1, 3)):
            _dent(img, rng, 1.0)
    elif label == 3:
        for _ in range(rng.integers(2, 4)):
            _dent(img, rng, 1.2)
        _crack(img, rng)
        if rng.random() < 0.7:
            _rust(img, rng)
        for _ in range(rng.integers(0, 3)):
            _scratch(img, rng)
    elif rng.random() < 0.15:
        # Clean panels sometimes carry a harmless reflection streak that looks like a scratch.
        _scratch(img, rng)
        img[:] = 0.5 * img + 0.5 * _panel(rng, s)
    return np.clip(img, 0, 1)


def synthetic_dataset(n_per_class, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in range(len(CLASSES)):
        for _ in range(n_per_class):
            X.append(synthetic_image(label, rng))
            y.append(label)
    X, y = np.array(X, dtype=np.float64), np.array(y)
    order = rng.permutation(len(y))
    return X[order], y[order]


def folder_dataset(root, seed, test_fraction=0.2):
    # Real images in root/<class name>/*.jpg|png, class names as in CLASSES; resized to 32x32.
    from PIL import Image

    X, y = [], []
    for label, name in enumerate(CLASSES):
        for path in sorted((Path(root) / name).glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                im = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                X.append(np.asarray(im, dtype=np.float64) / 255.0)
                y.append(label)
    if not X:
        raise SystemExit(f"No images found under {root}/<{'|'.join(CLASSES)}>/")
    X, y = np.array(X), np.array(y)
    order = np.random.default_rng(seed).permutation(len(y))
    X, y = X[order], y[order]
    cut = int(len(y) * (1 - test_fraction))
    return (X[:cut], y[:cut]), (X[cut:], y[cut:])


def load(data_dir=None, seed=2026):
    if data_dir:
        return folder_dataset(data_dir, seed)
    return synthetic_dataset(1500, seed), synthetic_dataset(300, seed + 1)
