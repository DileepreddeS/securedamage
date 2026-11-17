# SecureDamage

Privacy-preserving vehicle damage classification with homomorphic encryption (CKKS, OpenFHE).

A claimant encrypts a photo of a damaged panel on their own device. The insurer runs its damage
classifier on the ciphertext and returns an encrypted answer that only the claimant can decrypt.
The insurer never sees the photo or the prediction; the claimant never sees the model's weights.

CYB 599 Graduate Project, Part 3 (Implementation & Evaluation).

---

## Quick start

You need **Linux x86-64 with Python 3.12**, or **Docker**. On Windows, use WSL with Ubuntu 24.04
(see [Platform notes](#platform-notes)).

```bash
git clone <this repository>
cd securedamage
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh --quick
```

`./run.sh --quick` trains the models, runs the test suite, classifies one image end to end, and
evaluates 20 images per parameter set. It takes about 2 minutes. `./run.sh` with no options runs
the full evaluation reported in the paper (1200 + 300 + 300 encrypted inferences, about 25 minutes
on 2 CPU cores).

### With Docker (cleanest)

```bash
docker build -t securedamage .
docker run --rm securedamage                                          # same as ./run.sh --quick
docker run --rm securedamage --full                                   # the full evaluation
docker run --rm -v "$PWD/results:/app/results" securedamage --quick   # keep the results on the host
```

---

## What the pipeline computes

```
client (owns secret key)                          server (owns model weights)
------------------------                          ---------------------------
keygen, eval keys  ---- context, pk, relin key, rotation keys ---->  rebuild context from bytes
image 32x32x3
 -> normalize, im2col (in the clear, on the client)
 -> pack into 10 CKKS ciphertexts ---- query (10 ciphertexts) ---->  conv 5x5/s2, 3->5 channels
                                                                      act  z^2 + k z
                                                                      fc   980 -> 32
                                                                      act  z^2 + k z
                                                                      fc   32 -> 4
decrypt 4 logits, argmax  <---------- response (1 ciphertext) ----
```

- **Scheme:** CKKS (OpenFHE 1.6.1), multiplicative depth 5, no bootstrapping.
- **Model:** a small CNN with degree-2 polynomial activations (a least-squares fit to ReLU), trained
  in plain numpy. A ReLU version of the same network is the non-private baseline.
- **Packing:** one image uses 8192 slots. The convolution needs no rotations except a final
  3-step block sum; the dense layers use the GAZELLE "hybrid" diagonal method with baby-step /
  giant-step rotations. 18 rotation keys in total.
- **Wire protocol:** everything the server receives is serialized bytes. The server process
  rebuilds the context and evaluation keys from that bundle; it never holds a secret key.

## Parameter sets

| Set | Security | Scale | N | log2 Q |
|---|---|---|---|---|
| A | 128-bit classical | 2^40 | 16384 | 279 |
| B | 128-bit classical | 2^50 | 32768 | 330 |
| C | 192-bit classical | 2^40 | 32768 | 280 |

OpenFHE picks N from the HE Security Standard and the evaluation confirms it refuses N/2 for
every set.

---

## Repository layout

```
securedamage/
  __init__.py      class names, image size
  data.py          synthetic damage dataset; loader for real images in class folders
  model.py         numpy CNN: im2col, polynomial / ReLU activations, Adam training, save/load
  packing.py       slot layouts, diagonal method, and a numpy simulator of the exact HE algorithm
  he.py            parameter sets, Client (keys, encrypt, decrypt), Server (encrypted inference)
scripts/
  train.py         trains the poly (HE) model and the ReLU baseline -> models/, results/training.json
  demo.py          one image end to end, printing every message size
  evaluate.py      correctness + performance across parameter sets -> results/
tests/
  test_model.py    im2col vs direct convolution; analytic vs finite-difference gradients
  test_packing.py  slot algorithm == dense model (to 1e-10); activation folding; rotation key coverage
  test_he.py       encrypted vs plaintext logits; 128-bit check; bundle has no secret key; wrong key fails
run.sh             one command to reproduce everything
Dockerfile         pinned clean environment (Ubuntu 24.04, Python 3.12)
requirements.txt   pinned OpenFHE wheel and other dependencies
```

## Running pieces individually

```bash
python -m scripts.train                     # ~1 minute
python -m pytest -q                         # 9 tests, ~5 seconds
python -m scripts.demo --set A --index 3    # one encrypted inference, any parameter set / test image
python -m scripts.evaluate --sets A,B --images 50
```

## Outputs

`results/summary.md` is a readable table of every number in the report. `results/summary.json`
has the same data with full precision, `results/per_image_<set>.csv` has one row per encrypted
inference, and the figures are `results/fig_latency_breakdown.png`, `fig_error_hist.png` and
`fig_baseline.png`. Timings depend on your CPU; accuracy, agreement and message sizes do not.

---

## Data

By default the pipeline uses a **synthetic** dataset (`securedamage/data.py`): procedurally drawn
32x32 painted panels with scratches, dents, cracks and rust, in four severity classes
(6000 training / 1200 test images, fixed seed). It exists so that the code runs from a clean
environment with no downloads. It measures the cryptographic pipeline, not real-world damage
assessment quality.

To use real images, arrange them as

```
my_images/
  none/      *.jpg
  minor/     *.jpg
  moderate/  *.jpg
  severe/    *.jpg
```

and run `./run.sh --data-dir my_images`. Images are resized to 32x32 and split 80/20.
Detection datasets (bounding boxes) need to be cropped to damage regions first.

---

## Platform notes

The `openfhe` package on PyPI is a compiled build for **Linux x86-64, Python 3.12, glibc >= 2.38**
(Ubuntu 24.04 or newer). On other platforms `pip install` may succeed but `import openfhe` fails
with `No module named 'openfhe.openfhe'`.

- **Windows:** install WSL (`wsl --install -d Ubuntu-24.04` in an administrator PowerShell),
  then run the Quick start inside the Ubuntu terminal. In VS Code, use the WSL extension and
  "WSL: Connect to WSL".
- **macOS or older Linux (e.g. RHEL/Rocky 8-9, many HPC clusters):** use Docker, or Apptainer:
  `apptainer exec docker://ubuntu:24.04 ...`.
- **Memory:** about 2 GB. Rotation keys alone are 162 MB (set A) to 270 MB (sets B, C).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Python 3.12 is required` | Create the venv with `python3.12`. On Ubuntu 24.04 that is the default `python3`. |
| `No module named 'openfhe.openfhe'` | Wrong platform or Python version, see Platform notes. |
| `libgomp.so.1: cannot open shared object file` | `sudo apt install libgomp1` |
| `./run.sh: Permission denied` | `chmod +x run.sh` or run `bash run.sh` |
| Killed during evaluation | Not enough memory; close other programs or run `--sets A`. |
