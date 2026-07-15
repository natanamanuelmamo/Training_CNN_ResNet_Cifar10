# Plain CNN vs. Residual CNN — CIFAR-10

Implementation for Part 2 of the CNN/ResNet assignment: a plain CNN (Model 1)
and a CNN with hand-implemented residual blocks (Model 2), trained and
evaluated on CIFAR-10 under identical conditions.

## 1. What's implemented

- **`src/models.py`** — Both networks from scratch (no pretrained/imported
  ResNet). They share the *exact same skeleton* (stem → 3 stages → global
  average pool → FC), so the only difference between them is whether each
  pair of conv layers has a shortcut connection. This directly mirrors the
  controlled 18-vs-34-layer plain/residual experiment in the original paper
  (Section 4.2), just scaled down to CIFAR-10.
  - `PlainBlock`: conv-BN-ReLU → conv-BN-ReLU, no shortcut.
  - `ResidualBlock`: same two conv-BN layers, plus `H(x) = F(x) + shortcut(x)`.
    Uses an **identity** shortcut when shape doesn't change, and a **1×1
    projection** shortcut (learned) when channels/stride change — exactly the
    two cases described in the paper.
  - With `n=3` blocks per stage you get a 20-layer network (`6n+2`), i.e. a
    "ResNet-20"-style model, matching the smallest CIFAR variant in the paper.
- **`src/data.py`** — CIFAR-10 loading with a fixed, shared preprocessing/
  augmentation pipeline (random crop + horizontal flip + normalization) so
  both models train on identical data.
- **`src/train.py`** — One shared training loop (SGD + momentum + weight
  decay + step LR schedule) used for both models, so hyperparameters are
  guaranteed identical.
- **`src/utils.py`** — Parameter counting, loss/accuracy curve plots,
  confusion matrix plots, comparison table generation.
- **`main.py`** — Runs the whole experiment end-to-end and writes every
  deliverable into `outputs/`.

All of the code has already been smoke-tested (forward/backward pass, shape
checks, a full mini training+eval loop, and a residual-block identity-fallback
check) to catch bugs before you spend GPU time on the real run.

## 2. How to run

```bash
pip install -r requirements.txt
python main.py
```

That's it — `main.py` downloads CIFAR-10 automatically (via `torchvision`,
first run only, ~170 MB), trains both models back-to-back, and writes all
deliverables to `outputs/`.

### Useful flags

```bash
python main.py --epochs 5 --n 2          # quick smoke test (few minutes on GPU)
python main.py --epochs 30 --n 3         # default, good accuracy/time tradeoff
python main.py --epochs 60 --n 3         # closer to paper-quality ResNet-20 (~91-92% test acc)
python main.py --no-augment              # ablation: disable augmentation
```

- `--n` controls blocks per stage (`6n+2` layers total). `n=3` (20 layers) is
  the default and matches the paper's smallest CIFAR model.
- `--epochs` controls training length. The LR schedule automatically decays
  at 50% and 75% of total epochs.

### Recommended: run on Google Colab

This code was written and smoke-tested in a CPU-only sandbox; it runs
correctly but CPU training of ~20-layer CNNs for 30 epochs on 45,000 images
is slow (multiple hours). **A free Colab GPU runtime finishes the same run in
roughly 20-40 minutes.** Steps:

1. Upload this whole folder to Colab (or `git clone` if you push it to a repo).
2. Runtime → Change runtime type → GPU.
3. `!pip install -r requirements.txt` (torch/torchvision are usually
   preinstalled on Colab already).
4. `!python main.py`

## 3. Outputs produced (all in `outputs/`)

| File | Deliverable it satisfies |
|---|---|
| `comparison_loss_curves.png` | Training & validation loss curves |
| `comparison_accuracy_curves.png` | Training & validation accuracy curves |
| `PlainCNN_confusion_matrix.png`, `ResNetCNN_confusion_matrix.png` | Confusion matrix (per model) |
| `comparison_table.csv`, `comparison_table.md` | Comparison table (params, accuracy, time) |
| `PlainCNN_history.json`, `ResNetCNN_history.json` | Raw per-epoch metrics |
| `PlainCNN_weights.pt`, `ResNetCNN_weights.pt` | Trained model weights |
| `run_log.txt` | Full console log, including final test accuracy, parameter counts, and training time for both models |

## 4. Experimental setup (identical for both models)

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 (45k train / 5k val / 10k test split) |
| Preprocessing | Per-channel normalization (CIFAR-10 mean/std) |
| Augmentation | Random crop (32, padding=4) + random horizontal flip |
| Batch size | 128 |
| Optimizer | SGD, momentum=0.9, weight decay=5e-4 |
| LR schedule | Start at 0.1, ×0.1 at 50% and 75% of training |
| Epochs | 30 (default; configurable) |
| Loss | Cross-entropy |

## 5. Project structure

```
cnn_resnet_project/
├── main.py                # run this
├── requirements.txt
├── README.md
├── src/
│   ├── models.py           # PlainCNN, ResNetCNN, PlainBlock, ResidualBlock
│   ├── data.py              # CIFAR-10 loaders + transforms
│   ├── train.py             # shared training loop
│   └── utils.py              # plotting / metrics / tables
└── outputs/                # created after running main.py
```

## 6. Notes for the report (Part 1 tie-in)

- The parameter counts of the two models are almost identical (the residual
  network only has a few extra parameters, from the three 1×1 projection
  shortcuts) — so any accuracy/convergence gap you observe is attributable to
  the shortcut connections themselves, not to model capacity. This is the
  same logic the paper uses in its own plain-vs-residual comparison.
- If you increase `--n` (deeper networks) you should be able to reproduce,
  in miniature, the paper's central finding: the plain network's *training*
  accuracy eventually gets worse as depth increases (degradation problem),
  while the residual network keeps improving or holds steady.
