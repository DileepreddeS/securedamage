"""Train the HE-friendly model (polynomial activations) and the non-private ReLU baseline."""
import argparse
import json
import time
from pathlib import Path

from securedamage import data
from securedamage.model import Model

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", help="real images in <dir>/<none|minor|moderate|severe>/; default: synthetic")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    (Xtr, ytr), (Xte, yte) = data.load(args.data_dir)
    print(f"Dataset: {'folder ' + args.data_dir if args.data_dir else 'synthetic'}, "
          f"{len(ytr)} train / {len(yte)} test images")

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    report = {"dataset": args.data_dir or "synthetic", "n_train": len(ytr), "n_test": len(yte)}
    for act in ("relu", "poly"):
        print(f"\nTraining {act} model ({args.epochs} epochs)")
        model = Model(act, seed=0)
        start = time.perf_counter()
        model.train(Xtr, ytr, epochs=args.epochs)
        report[act] = {"train_seconds": time.perf_counter() - start,
                       "train_accuracy": model.accuracy(Xtr, ytr),
                       "test_accuracy": model.accuracy(Xte, yte)}
        model.save(ROOT / "models" / f"{act}.json")
        print(f"  test accuracy {report[act]['test_accuracy']:.4f}")

    report["poly_coefficients"] = list(Model("poly").coeffs)
    (ROOT / "results" / "training.json").write_text(json.dumps(report, indent=2))
    print("\nSaved models/relu.json, models/poly.json and results/training.json")


if __name__ == "__main__":
    main()
