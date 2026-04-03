# VI_using_LC

A Python research codebase for synthetic Gaussian mixture model (GMM) inference using a learned navigator, particle transport, and topology-aware diagnostics.

## What this repository contains

This repository trains and evaluates a **GMM navigator** that guides a particle-based inference procedure over streaming observation batches. The code combines:

- synthetic GMM problem generation,
- sequential batch assimilation,
- BAOAB-style particle dynamics,
- weighted particle inference,
- topology-aware diagnostics,
- evaluation and visualization scripts.

The main workflow is:

1. generate synthetic GMM episodes,
2. train a navigator on episodic inference tasks,
3. evaluate checkpoints on held-out problems,
4. visualize posterior structure and particle behavior.

## Repository layout

```text
VI_using_LC-main/
├── configs/
│   └── gmm.yaml
├── data/
│   ├── episode_sampler.py
│   └── gmm_problem.py
├── dynamics/
│   ├── baoab.py
│   ├── canonicalize.py
│   └── gmm_energy.py
├── fidelity/
│   └── casimir.py
├── models/
│   ├── gmm_embedder.py
│   └── gmm_navigator.py
├── topology/
│   ├── diagnostics.py
│   ├── filtration.py
│   ├── kde.py
│   └── phc.py
├── training/
│   ├── episode_trainer.py
│   └── losses.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── smoke_test.py
├── outputs/
├── outputs_modeB/
├── active_bayes_repo/
└── latent_transport/
```

## Main entrypoints

- `scripts/train.py` — train the navigator
- `scripts/evaluate.py` — evaluate a saved checkpoint
- `scripts/visualize.py` — generate plots and qualitative diagnostics
- `scripts/smoke_test.py` — run basic math / file / config sanity checks

## Environment setup

This repository is a **Python project**, so there is no CMake or compiled build step for the main codebase.

Recommended:

- Python 3.10+
- `venv` or `conda`
- optional CUDA-enabled PyTorch for GPU training

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies

The repository does not currently include a root-level `requirements.txt`, so install the core dependencies manually:

```bash
pip install torch numpy scipy matplotlib scikit-learn pyyaml
```

## Build / install instructions

There is no separate build step for the main project.

To get the repo ready to use:

```bash
git clone <your-repo-url>
cd VI_using_LC-main

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy scipy matplotlib scikit-learn pyyaml
```

That is the full setup for the main training / evaluation workflow.

## Device note

The default config in `configs/gmm.yaml` sets:

```yaml
device: cuda
```

So on a machine without CUDA, use `--device cpu` when running scripts, or edit the config and change `device: cpu`.

Example:

```bash
python scripts/train.py --config configs/gmm.yaml --output outputs/ --device cpu
```

## Smoke test

Before training, run the smoke test:

```bash
python scripts/smoke_test.py
```

This checks:

- GMM parameter dimension math,
- likelihood shape logic,
- union-find behavior,
- KDE bandwidth calculation,
- feature dimension math,
- BAOAB coefficients,
- ESS and resampling logic,
- expected file structure,
- config loading.

## Training

Train with the default configuration:

```bash
python scripts/train.py --config configs/gmm.yaml --output outputs/
```

For CPU-only training:

```bash
python scripts/train.py --config configs/gmm.yaml --output outputs/ --device cpu
```

Override the number of training episodes:

```bash
python scripts/train.py --config configs/gmm.yaml --output outputs/ --device cpu --n_episodes 100
```

Fix the number of GMM components for all episodes:

```bash
python scripts/train.py --config configs/gmm.yaml --output outputs/ --device cpu --K 4
```

Resume from a saved checkpoint:

```bash
python scripts/train.py \
  --config configs/gmm.yaml \
  --output outputs/ \
  --device cpu \
  --resume outputs/navigator_best.pt
```

### Expected training artifacts

Training writes checkpoints such as:

- `navigator_best.pt`
- `navigator_final.pt`
- intermediate checkpoints like `navigator_200.pt`

It also writes a loss history JSON file:

- `loss_history.json`

## Evaluation

Evaluate a trained checkpoint:

```bash
python scripts/evaluate.py \
  --checkpoint outputs/navigator_best.pt \
  --config configs/gmm.yaml \
  --n_test 100
```

CPU-only evaluation:

```bash
python scripts/evaluate.py \
  --checkpoint outputs/navigator_best.pt \
  --config configs/gmm.yaml \
  --n_test 100 \
  --device cpu
```

Write results to a specific JSON path:

```bash
python scripts/evaluate.py \
  --checkpoint outputs/navigator_best.pt \
  --config configs/gmm.yaml \
  --n_test 100 \
  --device cpu \
  --output outputs/eval_results.json
```

## Visualization

Generate plots for a trained checkpoint:

```bash
python scripts/visualize.py \
  --checkpoint outputs/navigator_best.pt \
  --config configs/gmm.yaml \
  --K 4 \
  --seed 42 \
  --output outputs/plots/
```

CPU-only visualization:

```bash
python scripts/visualize.py \
  --checkpoint outputs/navigator_best.pt \
  --config configs/gmm.yaml \
  --K 4 \
  --seed 42 \
  --output outputs/plots/ \
  --device cpu
```

## Configuration

The main experiment config is:

```bash
configs/gmm.yaml
```

Important sections:

- `problem` — GMM generation and streaming setup
- `particle` — number of particles and prior initialization scale
- `integrator` — BAOAB hyperparameters
- `navigator` — navigator architecture and control ranges
- `topology` — topology / filtration settings
- `training` — optimizer and training schedule
- `control` — control mode
- `evaluation` — evaluation settings

## Notes and caveats

- Run commands from the **repository root**.
- The scripts add the repository root to `sys.path`, so they are designed to be launched as `python scripts/<name>.py`.
- The repo currently has saved checkpoints in `outputs/` and `outputs_modeB/`, which can be used directly for evaluation and visualization.
- `active_bayes_repo/` appears to be a separate auxiliary subproject with its own `requirements.txt`; it is not required for the main `scripts/train.py` / `scripts/evaluate.py` / `scripts/visualize.py` workflow.
- `latent_transport/` also looks like a separate experiment and is not needed for the main GMM navigator workflow.

## Minimal quickstart

```bash
git clone <your-repo-url>
cd VI_using_LC-main

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy scipy matplotlib scikit-learn pyyaml

python scripts/smoke_test.py

python scripts/train.py --config configs/gmm.yaml --output outputs/ --device cpu

python scripts/evaluate.py \
  --checkpoint outputs/navigator_best.pt \
  --config configs/gmm.yaml \
  --n_test 50 \
  --device cpu

python scripts/visualize.py \
  --checkpoint outputs/navigator_best.pt \
  --config configs/gmm.yaml \
  --K 4 \
  --output outputs/plots/ \
  --device cpu
```
