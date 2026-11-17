"""Correctness and performance evaluation across parameter sets, plus the non-private baseline.

Writes results/summary.json, results/summary.md, results/per_image_<set>.csv and results/fig_*.png.
"""
import argparse
import csv
import gc
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import openfhe as fhe

from securedamage import data
from securedamage.he import Client, Server, PARAM_SETS, reset_key_cache, DEPTH
from securedamage.model import Model, im2col, preprocess

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STAGES = ["deserialize", "conv", "act1", "fc1", "act2", "fc2", "serialize"]
HE_STD_128 = {8192: 218, 16384: 438, 32768: 881}
HE_STD_192 = {8192: 152, 16384: 305, 32768: 611}


def plaintext_baseline(model, X, repeats):
    # Per-image latency of the plaintext model, one image at a time (as a query would arrive).
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        model.forward(im2col(preprocess(X[i:i + 1])))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def security_probe(pset, n):
    # Ask OpenFHE for the same parameters with half the ring dimension; its HE-Standard check should refuse.
    params = fhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(pset.scale_bits)
    params.SetFirstModSize(pset.first_mod_bits)
    params.SetBatchSize(8192)
    params.SetSecurityLevel(getattr(fhe, f"HEStd_{pset.security}_classic"))
    params.SetRingDim(n // 2)
    if pset.digits:
        params.SetNumLargeDigits(pset.digits)
    try:
        fhe.GenCryptoContext(params)
        return "accepted"
    except RuntimeError as e:
        return "rejected: " + str(e).split("):")[-1].strip()


def run_set(name, model, X, y, ref_logits, n_images):
    pset = PARAM_SETS[name]
    reset_key_cache()
    gc.collect()
    client = Client(pset)
    info = client.info()
    start = time.perf_counter()
    bundle = client.public_bundle()
    export_s = time.perf_counter() - start

    start = time.perf_counter()
    server = Server(bundle, model)
    server_setup_s = time.perf_counter() - start

    # Warm-up query: the server encodes its weight plaintexts at the right levels once, and keeps them.
    start = time.perf_counter()
    client.decrypt_logits(server.infer(client.encrypt_image(X[0])))
    first_query_s = time.perf_counter() - start

    rows = []
    for i in range(n_images):
        server.timer.t = {}
        t0 = time.perf_counter()
        query = client.encrypt_image(X[i])
        t1 = time.perf_counter()
        response = server.infer(query)
        t2 = time.perf_counter()
        logits = client.decrypt_logits(response)
        t3 = time.perf_counter()
        ref = ref_logits[i]
        top2 = np.sort(ref)[-2:]
        rows.append({
            "index": i, "label": int(y[i]),
            "plain_pred": int(ref.argmax()), "he_pred": int(logits.argmax()),
            "max_abs_err": float(np.max(np.abs(logits - ref))),
            "max_rel_err": float(np.max(np.abs(logits - ref)) / np.max(np.abs(ref))),
            "margin": float(top2[1] - top2[0]),
            "encrypt_s": t1 - t0, "server_s": t2 - t1, "decrypt_s": t3 - t2,
            **{f"stage_{s}_s": server.timer.t.get(s, 0.0) for s in STAGES},
            "upload_bytes": sum(map(len, query)), "download_bytes": len(response),
        })
        if (i + 1) % 50 == 0 or i + 1 == n_images:
            print(f"    {i + 1}/{n_images} images")

    with open(RESULTS / f"per_image_{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    col = lambda k: np.array([r[k] for r in rows])
    stat = lambda k: {"mean": float(col(k).mean()), "std": float(col(k).std()), "min": float(col(k).min()),
                      "max": float(col(k).max()), "median": float(np.median(col(k)))}
    n = info["N"]
    table = HE_STD_128 if pset.security == "128" else HE_STD_192
    summary = {
        "label": pset.label, **info,
        "he_std_max_log2_QP": table[n],
        "half_ring_dimension": security_probe(pset, n),
        "client_setup_s": {k: v for k, v in client.timer.t.items()}, "bundle_export_s": export_s,
        "server_setup_s": server_setup_s, "first_query_s": first_query_s,
        "bundle_bytes": {k: len(v) for k, v in bundle.items()},
        "n_images": n_images,
        "argmax_agreement": float((col("plain_pred") == col("he_pred")).mean()),
        "he_accuracy": float((col("he_pred") == col("label")).mean()),
        "plain_accuracy_same_images": float((col("plain_pred") == col("label")).mean()),
        "max_abs_err": stat("max_abs_err"), "max_rel_err": stat("max_rel_err"),
        "min_margin": float(col("margin").min()),
        "encrypt_s": stat("encrypt_s"), "server_s": stat("server_s"), "decrypt_s": stat("decrypt_s"),
        "stages_s": {s: float(col(f"stage_{s}_s").mean()) for s in STAGES},
        "upload_bytes": int(col("upload_bytes")[0]), "download_bytes": int(col("download_bytes")[0]),
    }
    del server, client, bundle
    reset_key_cache()
    gc.collect()
    return summary


def figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = list(summary["param_sets"])
    labels = [f"Set {s}\n(N={summary['param_sets'][s]['N']}, {summary['param_sets'][s]['security']}-bit,\n"
              f"scale 2^{summary['param_sets'][s]['scale_bits']})" for s in sets]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#8C8C8C"]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bottom = np.zeros(len(sets))
    for stage, c in zip(STAGES, colors):
        vals = np.array([summary["param_sets"][s]["stages_s"][stage] for s in sets])
        ax.bar(labels, vals, bottom=bottom, label=stage, color=c, width=0.55)
        bottom += vals
    ax.set_ylabel("server time per image (s)")
    ax.set_title("Server-side latency breakdown by layer")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_latency_breakdown.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for s, c in zip(sets, colors):
        errs = []
        with open(RESULTS / f"per_image_{s}.csv") as f:
            errs = [float(r["max_abs_err"]) for r in csv.DictReader(f)]
        ax.hist(np.log10(errs), bins=30, alpha=0.6, color=c, label=f"Set {s} (scale 2^{summary['param_sets'][s]['scale_bits']})")
    ax.set_xlabel("log10 of max |encrypted logit - plaintext logit| per image")
    ax.set_ylabel("images")
    ax.set_title("CKKS approximation error of the returned logits")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_error_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    names = ["Plaintext ReLU\n(non-private)", "Plaintext poly\n(non-private)"] + [f"Encrypted\nSet {s}" for s in sets]
    vals = [summary["baseline"]["relu_latency_s"], summary["baseline"]["poly_latency_s"]]
    vals += [summary["param_sets"][s]["encrypt_s"]["mean"] + summary["param_sets"][s]["server_s"]["mean"]
             + summary["param_sets"][s]["decrypt_s"]["mean"] for s in sets]
    bars = ax.bar(names, vals, color=["#8C8C8C", "#8C8C8C"] + colors[:len(sets)], width=0.55)
    ax.set_yscale("log")
    ax.set_ylim(min(vals) / 2, max(vals) * 4)
    ax.set_ylabel("end-to-end latency per image (s, log scale)")
    ax.set_title("Encrypted inference vs. non-private baseline")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v * 1000:.2f} ms" if v < 0.1 else f"{v:.2f} s",
                ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_baseline.png", dpi=200)
    plt.close(fig)


def markdown(summary):
    MB = 1024 * 1024
    lines = ["# SecureDamage evaluation summary", "",
             f"Machine: {summary['machine']}", "",
             "## Non-private baseline (plaintext numpy)", "",
             f"- ReLU model: test accuracy {summary['baseline']['relu_accuracy']:.4f}, "
             f"latency {summary['baseline']['relu_latency_s'] * 1000:.3f} ms/image",
             f"- Polynomial model: test accuracy {summary['baseline']['poly_accuracy']:.4f}, "
             f"latency {summary['baseline']['poly_latency_s'] * 1000:.3f} ms/image",
             f"- Query size: {summary['baseline']['image_bytes']} bytes (raw 32x32x3 uint8 image)", "",
             "## Encrypted inference", "",
             "| | " + " | ".join(f"Set {s}" for s in summary["param_sets"]) + " |",
             "|---|" + "---|" * len(summary["param_sets"])]
    ps = summary["param_sets"]
    rows = [
        ("Parameters", lambda d: d["label"]),
        ("Ring dimension N", lambda d: str(d["N"])),
        ("log2 Q", lambda d: f"{d['log2_Q']:.0f}"),
        ("OpenFHE at N/2", lambda d: "rejected" if d["half_ring_dimension"].startswith("rejected") else "ACCEPTED"),
        ("Images evaluated", lambda d: str(d["n_images"])),
        ("Argmax agreement with plaintext", lambda d: f"{d['argmax_agreement'] * 100:.2f}%"),
        ("Encrypted accuracy", lambda d: f"{d['he_accuracy'] * 100:.2f}%"),
        ("Max abs logit error (worst image)", lambda d: f"{d['max_abs_err']['max']:.2e}"),
        ("Median abs logit error", lambda d: f"{d['max_abs_err']['median']:.2e}"),
        ("Smallest plaintext logit margin", lambda d: f"{d['min_margin']:.3f}"),
        ("Client encrypt (s)", lambda d: f"{d['encrypt_s']['mean']:.3f} ± {d['encrypt_s']['std']:.3f}"),
        ("Server evaluate (s)", lambda d: f"{d['server_s']['mean']:.3f} ± {d['server_s']['std']:.3f}"),
        ("Client decrypt (s)", lambda d: f"{d['decrypt_s']['mean'] * 1000:.2f} ms"),
        ("Upload per query (MB)", lambda d: f"{d['upload_bytes'] / MB:.2f}"),
        ("Download per query (MB)", lambda d: f"{d['download_bytes'] / MB:.2f}"),
        ("Rotation keys (MB, once)", lambda d: f"{d['bundle_bytes']['rotation_keys'] / MB:.1f}"),
        ("Relinearization key (MB, once)", lambda d: f"{d['bundle_bytes']['relin_key'] / MB:.1f}"),
        ("Public key (MB, once)", lambda d: f"{d['bundle_bytes']['public_key'] / MB:.2f}"),
        ("Client key generation (s, once)", lambda d: f"{d['client_setup_s']['keygen'] + d['client_setup_s']['evalkeygen']:.2f}"),
    ]
    for label, fmt in rows:
        lines.append(f"| {label} | " + " | ".join(fmt(ps[s]) for s in ps) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", default="A,B,C")
    ap.add_argument("--images", type=int, default=200, help="test images per parameter set")
    ap.add_argument("--images-A", type=int, default=None, help="override for set A (e.g. the full test set)")
    ap.add_argument("--data-dir")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    poly = Model.load(ROOT / "models" / "poly.json")
    relu = Model.load(ROOT / "models" / "relu.json")
    _, (Xte, yte) = data.load(args.data_dir)
    ref = poly.forward(im2col(preprocess(Xte)))

    summary = {
        "machine": f"{platform.processor() or platform.machine()}, {os.cpu_count()} CPU threads, "
                   f"Python {platform.python_version()}, OpenFHE {fhe.__version__ if hasattr(fhe, '__version__') else 'python'}",
        "baseline": {
            "relu_accuracy": relu.accuracy(Xte, yte), "poly_accuracy": poly.accuracy(Xte, yte),
            "relu_latency_s": plaintext_baseline(relu, Xte, 300),
            "poly_latency_s": plaintext_baseline(poly, Xte, 300),
            "image_bytes": 32 * 32 * 3, "n_test": len(yte),
        },
        "param_sets": {},
    }
    print(f"Plaintext baseline: ReLU {summary['baseline']['relu_accuracy']:.4f}, "
          f"poly {summary['baseline']['poly_accuracy']:.4f} test accuracy")

    for name in args.sets.split(","):
        n = min(args.images_A if (name == "A" and args.images_A) else args.images, len(yte))
        print(f"Parameter set {name} ({PARAM_SETS[name].label}): {n} images")
        summary["param_sets"][name] = run_set(name, poly, Xte, yte, ref, n)

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / "summary.md").write_text(markdown(summary))
    figures(summary)
    print("\n" + markdown(summary))
    print("Wrote results/summary.json, results/summary.md, results/per_image_*.csv, results/fig_*.png")


if __name__ == "__main__":
    main()
