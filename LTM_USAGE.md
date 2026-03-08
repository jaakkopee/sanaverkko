# LTM usage (`sv_ltm.py`)

This document focuses on CLI usage and parameters for training a long-term memory model (`.svltm`) with `sv_ltm.py`.

## Quick start

Train from a text file:

```bash
python sv_ltm.py --input chicken_qabalah.txt --output chicken_qabalah --epochs 20 --context 3
```

If `--output` does not end with `.svltm`, the suffix is added automatically.

## CLI parameters

## Required

- `--input`
  - Path to input text file used for training.
  - The file is tokenized to lowercase words matching `[a-zåäö]+`.

- `--output`
  - Output model file path.
  - If missing extension, `.svltm` is appended.

## Core training

- `--epochs` (int, default: `10`)
  - Number of full training passes over the dataset.

- `--lr` (float, default: `0.02`)
  - Learning rate for optimizer step updates.

- `--context` (int, default: `3`)
  - Number of previous words used as context for next-word prediction.
  - This should match your intended runtime behavior in `sanaVerkkoCore.py`.

## MLP head size

- `--hidden1` (int, default: `1024`)
  - First hidden layer width.

- `--hidden2` (int, default: `512`)
  - Second hidden layer width.

Larger hidden sizes improve capacity but increase model size and memory/CPU cost.

## Vocabulary and embedding/CNN

- `--min-word-count` (int, default: `1`)
  - Minimum frequency threshold for a word to stay in vocabulary.
  - Increase this to reduce vocabulary size and model size.

- `--embedding-dim` (int, default: `32`)
  - Character embedding dimension.

- `--filters` (int, default: `64`)
  - Number of convolution filters per width in Char-CNN.

- `--max-word-len` (int, default: `24`)
  - Maximum word length used by the Char-CNN encoder.
  - Longer words are truncated.

## Batch and reproducibility

- `--batch-size` (int, default: `64`)
  - Batch size for training steps.

- `--device` (`auto|cpu|mps|cuda`, default: `auto`)
  - Selects training backend device.
  - `auto` prefers `cuda`, then `mps`, then `cpu`.
  - GPU acceleration is used when `torch` is installed and selected device is available.

- `--seed` (int, default: `1234`)
  - RNG seed for deterministic initialization/order (same data + same params).

## Recommended presets

## Fast iteration

```bash
python sv_ltm.py \
  --input input.txt \
  --output fast_test.svltm \
  --epochs 5 \
  --hidden1 512 \
  --hidden2 256 \
  --filters 32 \
  --device auto
```

## Balanced baseline

```bash
python sv_ltm.py \
  --input kalevala.txt \
  --output kalevala_balanced.svltm \
  --epochs 20 \
  --lr 0.02 \
  --context 3 \
  --hidden1 1024 \
  --hidden2 512 \
  --embedding-dim 32 \
  --filters 64 \
  --batch-size 64 \
  --device auto
```

## Larger model (targeting >20MB)

```bash
python sv_ltm.py \
  --input chicken_qabalah.txt \
  --output chicken_qabalah_large.svltm \
  --epochs 30 \
  --context 3 \
  --hidden1 2048 \
  --hidden2 1024 \
  --embedding-dim 64 \
  --filters 128 \
  --batch-size 128 \
  --device auto
```

## GPU notes

- GPU training/inference support is optional and uses PyTorch when available.
- macOS (Apple Silicon): use `--device mps` (or `--device auto`).
- NVIDIA: use `--device cuda`.
- If PyTorch or the requested device is unavailable, training falls back to CPU path.

## Model size guidance

Model size is mostly affected by:
- vocabulary size,
- `hidden1`, `hidden2`,
- Char-CNN dimensions (`embedding-dim`, `filters`),
- cached per-word feature vectors.

To reduce size:
- increase `--min-word-count`,
- reduce hidden sizes,
- reduce `--embedding-dim` and/or `--filters`.

## Runtime usage in SanaVerkko

1. Open app: `python sanaVerkkoCore.py`
2. In control window:
   - click **Load long term memory** and select `.svltm`
   - enable **Use long term memory**
3. LTM then influences candidate ranking during word replacement.

## Common errors

- `ValueError: Not enough words for requested context size`
  - Input corpus is too small for selected `--context`.

- `ValueError: Vocabulary too small after filtering`
  - Your `--min-word-count` is too high for corpus size.

- Low-quality predictions
  - Increase corpus size and epochs.
  - Try lower `--min-word-count`.
  - Increase model capacity (`hidden1/hidden2`, `filters`).

## Notes

- `sv_ltm.py` trains a word-level next-token predictor with a char-based word encoder.
- Training uses NumPy by default, with optional PyTorch acceleration (`mps`/`cuda`) when available.
- Runtime loading in app requires the same Python environment to have compatible dependencies.
