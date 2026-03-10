# Active Bayes Teacher + Latent pH Transport Repo

A research scaffold for **active sequential assimilation** with:

- a labelled feature encoder into latent space,
- an **active query policy** over candidate observations,
- a training-time **Bayes teacher** that scores candidate queries using ground-truth episode information,
- a latent **port-Hamiltonian transport** update for the posterior cloud,
- cluster-aware summaries, memory, and visualization panels.

## Layout

- `configs/` experiment configs for synthetic GMM and external labelled NPZ data
- `data_generation/` active toy episode builder and external NPZ loader
- `models/` feature projector, latent energy, query policy, active assimilator
- `training/` projector training and active assimilation training loop
- `evaluation/` metrics for query accuracy, clustering quality, ESS, transport gap
- `visualization/` active-query, mode-identification, transport, and alignment panels

## Quick start

```bash
python generate_data.py --config configs/active_gmm.yaml
python train.py --config configs/active_gmm.yaml
python eval.py --config configs/active_gmm.yaml --checkpoint outputs_active/checkpoints/best.pt
python visualize.py --config configs/active_gmm.yaml --checkpoint outputs_active/checkpoints/best.pt
```

## External labelled data

Prepare an NPZ with keys like:

- `x_train`, `y_train`
- `x_val`, `y_val`
- `x_test`, `y_test`
- optional: `z_train`, `z_val`, `z_test`

If `z_*` are omitted, the code builds a PCA proxy latent for training-time teacher scoring and visualization.

Then use `configs/external_labeled.yaml` and set `data.external_npz_path`.

## Evaluation

The main metrics are:

- **query accuracy** against the Bayes teacher,
- **purity / ARI / NMI** of inferred basins on selected observations,
- **ESS** of the posterior cloud,
- **transport gap** between student and teacher cloud centroids,
- **teacher sensitivity** across candidate actions.

## Visualization

For 2D latent states, the code renders energy contours, particles, teacher cloud, queried batches, candidate anchors, and selected-vs-teacher actions.
For higher-dimensional latent states, it falls back to PCA projection for the scatter-based panels.
