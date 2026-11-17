#!/usr/bin/env bash
# Reproduce everything: train both models, run the test suite, run the evaluation.
#   ./run.sh           full evaluation (about 25 minutes on 2 CPU cores); same as --full
#   ./run.sh --quick   20 images per parameter set (about 2 minutes)
#   ./run.sh --data-dir /path/to/images   use real images instead of the synthetic dataset
set -euo pipefail
cd "$(dirname "$0")"

QUICK=0
DATA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --full) QUICK=0; shift ;;
    --data-dir) DATA_ARGS=(--data-dir "$2"); shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

python - <<'EOF'
import sys
if sys.version_info[:2] != (3, 12):
    sys.exit(f"Python 3.12 is required (found {sys.version.split()[0]}); see README.md")
try:
    import openfhe  # noqa: F401
except ImportError as e:
    sys.exit(f"OpenFHE failed to import ({e}); run: pip install -r requirements.txt  (Linux x86-64 only, see README.md)")
EOF

echo "== 1/4 Training models =="
python -m scripts.train "${DATA_ARGS[@]}"

echo; echo "== 2/4 Test suite =="
python -m pytest -q

echo; echo "== 3/4 End-to-end demo (one image) =="
python -m scripts.demo "${DATA_ARGS[@]}"

echo; echo "== 4/4 Evaluation =="
if [[ $QUICK -eq 1 ]]; then
  python -m scripts.evaluate --images 20 "${DATA_ARGS[@]}"
else
  python -m scripts.evaluate --images 300 --images-A 1200 "${DATA_ARGS[@]}"
fi

echo; echo "Done. Results are in results/ (summary.md, summary.json, per_image_*.csv, fig_*.png)."
