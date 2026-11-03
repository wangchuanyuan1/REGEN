# REGEN: Inferring Time-Resolved Gene Regulatory Networks from Global Network Supervision

<p align="center">
  <img src="figures/REGEN_LOGO.png" alt="REGEN Logo" width="320">
</p>

REGEN (Regulatory-network Generation) reconstructs time-resolved gene regulatory networks (GRNs) from single-cell RNA expression data ordered by chronological time or pseudotime, using global network supervision. By integrating RNA expression features, latent variable modeling, Bernoulli diffusion, and graph message passing, REGEN captures temporal changes in regulatory interactions while maintaining consistency between dynamic and global networks.

This repository provides the model implementation, unified training and inference interfaces, and complete runnable examples using simulated and real single-cell data.

## Key Features

- **Global supervision for dynamic reconstruction**: Learn regulatory interactions within each time window from global candidate-edge labels.
- **Multilevel expression modeling**: Integrate directional RNA features, node latent variables, diffusion states, and graph structure.
- **Consistency across dynamic and global networks**: Combine predictions from individual time windows into a global network through probability aggregation.
- **A streamlined research workflow**: Use command-line training and inference with edge-level CSV outputs for downstream network analysis and visualization.

## Repository Structure

```text
REGEN_GitHub/
├── src/regen/                # Model, data interfaces, training, and inference
├── examples/simulation/      # Simulated data with labels
├── examples/real_a549_dex/   # GSE117089 A549 DEX real-data subset
├── figures/                 # REGEN Logo
├── requirements.txt
├── LICENSE                  # MIT License
└── README.md
```

The `examples/` directory demonstrates the complete workflow from data input to network prediction. The simulated example includes labels derived from its generative mechanism. The real-data example provides expression data and candidate edges from GSE117089 A549 DEX; its `labels.csv` contains labels aggregated from the window-level GRNs released with the original data, illustrating the data organization for global network supervision.

## Installation

Python 3.8+ is recommended:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd REGEN_GitHub
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For GPU support, install the PyTorch build matching your CUDA environment by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Quick Start

Simulated data:

```bash
regen train \
  --input examples/simulation \
  --checkpoint outputs/simulation.pt \
  --windows 4 --epochs 30 --device cpu

regen infer \
  --input examples/simulation \
  --checkpoint outputs/simulation.pt \
  --output outputs/simulation_predictions.csv --device cpu
```

Real-data example:

```bash
regen train \
  --input examples/real_a549_dex \
  --checkpoint outputs/a549_dex.pt \
  --windows 4 --epochs 30 --device cpu

regen infer \
  --input examples/real_a549_dex \
  --checkpoint outputs/a549_dex.pt \
  --output outputs/a549_dex_predictions.csv --device cpu
```

Training checkpoints and predictions are saved under `outputs/` for convenient experiment management.

## Input Format

Organize each input directory as follows:

- `expression.csv`: Rows represent cells. The first column is `cell_id`, and the remaining columns contain gene expression values.
- `time.csv`: Two columns, `cell_id,pseudotime`.
- `candidates.csv` (optional): Two columns, `source,target`. When omitted, all directed gene pairs without self-loops are used automatically.
- `labels.csv` (required for training): Three columns, `source,target,label`, with binary labels of 0 or 1.

Inference produces an edge-level CSV containing `source`, `target`, a global probability, and dynamic probabilities for each time window. The global probability is aggregated from window-level probabilities as `1 - ∏(1 - p_window)`, ensuring exact consistency between the dynamic and global GRNs.

## Model

REGEN-HZBM comprises four core components: directional RNA features (H), Gaussian node latent variables (Z), Bernoulli diffusion-state training (B), and RNA graph message passing over all candidate edges (M). The model uses global candidate-edge labels as supervision to learn time-resolved regulatory networks.

## Data Source

The real-data example is drawn from the public single-cell dataset GSE117089 (A549 DEX). It provides a ready-to-run subset of expression data and candidate edges for exploring REGEN's training and inference workflow. Data use follows the terms of GEO and the original study.

## Cite the paper

If REGEN supports your research, please cite the associated manuscript:

> **REGEN: Inferring Time-Resolved Gene Regulatory Networks from Global Network Supervision**  
> Manuscript under review.

## MIT License

The REGEN code is released under the [MIT License](LICENSE), permitting free use, modification, and distribution.
