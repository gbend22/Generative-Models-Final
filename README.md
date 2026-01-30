# Generative Models on CIFAR-10

A PyTorch-based project implementing and comparing two generative modeling approaches on the CIFAR-10 dataset: **Noise Conditional Score Network (NCSN)** and **Beta-VAE**. Experiment tracking is handled through [Weights & Biases](https://wandb.ai/).

## Team

| Member                | GitHub Handle |
|-----------------------|--------------|
| Giorgi Bendianishvili | `gbend22`    |
| Givi Modebadze        | `gmode`      |

## Project Structure

```
Generative-Models-Final/
├── configs/
│   ├── ncsn_config.py            # NCSN hyperparameters and settings
│   └── beta_vae_config.py        # Beta-VAE hyperparameters and settings
├── src/
│   ├── data.py                   # CIFAR-10 data loading utilities
│   ├── utils.py                  # EMA helper, sigma schedule, checkpointing
│   ├── models/
│   │   ├── ncsn.py               # Score network (U-Net architecture)
│   │   ├── beta_vae.py           # Beta-VAE encoder/decoder
│   │   └── refineNet.py          # Alternative RefineNet architecture for NCSN
│   └── training/
│       ├── train_ncsn.py         # NCSN training loop
│       ├── train_beta_vae.py     # Beta-VAE training loop
│       ├── loss.py               # Denoising score matching loss
│       ├── loss_beta_vae.py      # Reconstruction + KL divergence loss
│       ├── sampling.py           # Annealed Langevin dynamics sampling
│       └── sampling_beta_vae.py  # VAE sampling, reconstruction, latent traversal
├── main.py                       # Entry point: NCSN training
├── main_beta_vae.py              # Entry point: Beta-VAE training
├── requirements.txt
└── README.md
```

## Implemented Models

### NCSN (Noise Conditional Score Network)

Based on [Song & Ermon (2019)](https://arxiv.org/abs/1907.05600). Learns to estimate the score (gradient of log-probability) of the data distribution at multiple noise levels, then generates samples via **annealed Langevin dynamics**.

- **Architecture:** U-Net encoder-decoder with Gaussian Fourier projection for noise conditioning, GroupNorm, SiLU activations, and skip connections.
- **Loss:** Denoising score matching — the network predicts the noise direction added to each sample, weighted by sigma at each noise scale.
- **Sampling:** Iterates through 50 geometrically-spaced noise levels (sigma 1.0 to 0.01), running 100 Langevin dynamics steps per level.

**Default hyperparameters** (see `configs/ncsn_config.py`):
| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Learning rate | 1e-3 |
| Epochs | 200 |
| Noise levels | 50 |
| Langevin steps per level | 100 |
| Step LR | 2e-5 |

### Beta-VAE

Based on [Higgins et al. (2017)](https://openreview.net/forum?id=Sy2fzU9gl). A variational autoencoder with a tunable beta parameter that controls the trade-off between reconstruction quality and latent disentanglement.

- **Architecture:** 4-layer convolutional encoder (64 -> 128 -> 256 -> 512) and mirrored transposed-convolution decoder, with batch normalization.
- **Loss:** MSE reconstruction loss + beta-weighted KL divergence against a standard normal prior.
- **Sampling:** Decode from samples drawn from N(0, I). Supports latent traversals and interpolations.

**Default hyperparameters** (see `configs/beta_vae_config.py`):
| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Learning rate | 1e-4 |
| Epochs | 150 |
| Latent dimension | 128 |
| Beta | 1.0 |

## Branch Strategy

This project uses feature branches for individual experiments and model work. Each branch focuses on a specific improvement or set of changes.

| Branch | Owner   | Purpose |
|--------|---------|---------|
| `main` | shared  | Stable baseline code |
| `fix-for-ncsn` | gbend22 | Config and setup fixes for the NCSN pipeline |
| `beta_vae` | gmode   | Implementation and tuning of the Beta-VAE model |
| `gmode_ncsn` | gmode   | NCSN experiments — step learning rate adjustments |
| `gbend_ncsn` | gbend22 | NCSN experiments — normalization fixes |

When starting a new experiment, create a branch off of the latest shared work. Merge back into `main` once results are verified.

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended; CPU training is supported but slow)
- A [Weights & Biases](https://wandb.ai/) account for experiment tracking

### Installation

```bash
git clone <repo-url>
cd Generative-Models-Final
pip install -r requirements.txt
wandb login
```

### Training

**NCSN:**
```bash
python main.py
```

**Beta-VAE:**
```bash
python main_beta_vae.py
```

The Beta-VAE entry point supports command-line overrides:
```bash
python main_beta_vae.py --beta 4.0 --epochs 200 --latent_dim 256 --lr 5e-5
```

### Checkpoints & Logging

- Checkpoints are saved every 10 epochs to the `checkpoints/` directory.
- Sample images are generated every 5 epochs and logged to WandB.
- All training metrics (loss curves, KL divergence, reconstruction loss) are tracked in the WandB project `cifar10-generative-models`.

---

## Models & Results

NCSN Best Loss And Model - 288.3 ncsn-baseline-higher-scales.

wandb for NCSN(https://wandb.ai/gbend22-free-university-of-tbilisi-/cifar10-generative-models/overview)

---

## References

- Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
- Higgins, I., et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR.
