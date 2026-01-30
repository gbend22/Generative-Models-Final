# Generative Models on CIFAR-10

A PyTorch-based project implementing and comparing three generative modeling approaches on the CIFAR-10 dataset: **Noise Conditional Score Network (NCSN)**, **Beta-VAE**, and **Importance Weighted Autoencoder (IWAE)**. Experiment tracking is handled through [Weights & Biases](https://wandb.ai/).

## Team

| Member                | GitHub Handle |
|-----------------------|--------------|
| Giorgi Bendianishvili | `gbend22`    |
| Givi Modebadze        | `Givi-Modebadze`      |

## Project Structure

```
Generative-Models-Final/
├── configs/
│   ├── ncsn_config.py            # NCSN hyperparameters and settings
│   ├── beta_vae_config.py        # Beta-VAE hyperparameters and settings
│   └── iwae_config.py            # IWAE hyperparameters and settings
├── src/
│   ├── data.py                   # CIFAR-10 data loading utilities
│   ├── utils.py                  # EMA helper, sigma schedule, checkpointing
│   ├── models/
│   │   ├── ncsn.py               # Score network (U-Net architecture)
│   │   ├── beta_vae.py           # Beta-VAE encoder/decoder
│   │   ├── iwae.py               # IWAE model (reuses Beta-VAE encoder/decoder)
│   │   └── refineNet.py          # Alternative RefineNet architecture for NCSN
│   └── training/
│       ├── train_ncsn.py         # NCSN training loop
│       ├── train_beta_vae.py     # Beta-VAE training loop
│       ├── train_iwae.py         # IWAE training loop
│       ├── loss.py               # Denoising score matching loss
│       ├── loss_beta_vae.py      # Reconstruction + KL divergence loss
│       ├── loss_iwae.py          # Importance-weighted ELBO loss
│       ├── sampling.py           # Annealed Langevin dynamics sampling
│       ├── sampling_beta_vae.py  # VAE sampling, reconstruction, latent traversal
│       └── sampling_iwae.py      # IWAE sampling and best-weight reconstruction
├── main.py                       # Entry point: NCSN training
├── main_beta_vae.py              # Entry point: Beta-VAE training
├── main_iwae.py                  # Entry point: IWAE training
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

### IWAE (Importance Weighted Autoencoder)

Based on [Burda, Grosse & Salakhutdinov (2016)](https://arxiv.org/abs/1509.00519). Uses the same encoder/decoder architecture as the VAE but trains with a strictly tighter log-likelihood lower bound derived from importance weighting.

- **Architecture:** Same convolutional encoder/decoder as Beta-VAE (shared implementation). The difference is purely in the training objective.
- **Loss:** Instead of the standard ELBO (single-sample bound), IWAE draws k samples from the encoder and computes the importance-weighted bound: `L_k = E[log(1/k * sum w_i)]` where `w_i = p(x, z_i) / q(z_i | x)`. Uses `logsumexp` for numerical stability.
- **Sampling:** Identical to VAE — decode from samples drawn from N(0, I). Additionally supports best-weight reconstruction (selecting the sample with the highest importance weight).

**Default hyperparameters** (see `configs/iwae_config.py`):
| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 1e-4 |
| Epochs | 150 |
| Latent dimension | 128 |
| Importance samples (k) | 5 |

## Branch Strategy

This project uses feature branches for individual experiments and model work. Each branch focuses on a specific improvement or set of changes.

| Branch | Owner   | Purpose |
|--------|---------|---------|
| `main` | shared  | Stable baseline code |
| `fix-for-ncsn` | gbend22 | Config and setup fixes for the NCSN pipeline |
| `beta_vae` | gmode   | Implementation and tuning of the Beta-VAE model |
| `gmode_ncsn` | gmode   | NCSN experiments — step learning rate adjustments |
| `gbend_ncsn` | gbend22 | NCSN experiments — normalization fixes |
| `gmode_iwae` | gmode   | Implementation and experiments for IWAE |

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

**IWAE:**
```bash
python main_iwae.py
```

The IWAE entry point supports command-line overrides:
```bash
python main_iwae.py --k 50 --epochs 150 --batch_size 8
```

### Checkpoints & Logging

- Checkpoints are saved every 10 epochs to the `checkpoints/` directory.
- Sample images are generated every 5 epochs and logged to WandB.
- All training metrics (loss curves, KL divergence, reconstruction loss) are tracked in the WandB project `cifar10-generative-models`.

---

## Results

### NCSN

Best loss: **288.3** (run: `ncsn-baseline-higher-scales`)

WandB Experiments: [NCSN experiments](https://wandb.ai/gbend22-free-university-of-tbilisi-/cifar10-generative-models/overview)

WandB Report: [NCSN Report](https://wandb.ai/gbend22-free-university-of-tbilisi-/cifar10-generative-models/reports/Noise-Conditional-Score-Networks-on-CIFAR-10--VmlldzoxNTc5NTc2OQ)

### Beta-VAE

All runs trained for **200 epochs** on CIFAR-10 with latent dimension 128. Three beta values were compared:

| Run | Beta | Total Loss | Recon Loss | KL Loss |
|-----|------|-----------|------------|---------|
| `beta-vae-beta0.5` | 0.5 | 54.62 | 36.27 | 36.71 |
| `beta-vae-beta1.0` | 1.0 | 72.31 | 49.22 | 23.09 |
| `beta-vae-beta2.0` | 2.0 | 87.96 | 62.15 | 12.90 |

**Observations:**
- Lower beta produces better reconstruction (lower recon loss) at the cost of less structured latent space (higher KL).
- Higher beta forces more disentanglement (lower KL) but sacrifices reconstruction quality.
- Generated samples are blurry across all beta values, which is a known limitation of VAEs on CIFAR-10.

WandB: [Beta-VAE & IWAE experiments](https://wandb.ai/gmode-free-university-of-tbilisi-/cifar10-generative-models)

### IWAE

Two configurations were tested to demonstrate the effect of increasing the number of importance samples k:

| Run | k | Epochs | Train Loss | Recon Loss | KL Loss |
|-----|---|--------|-----------|------------|---------|
| `iwae-k5` | 5 | 150 | 44.59 | 68.65 | 12.23 |
| `iwae-k50` | 50 | 50 | 43.35 | 79.49 | 9.37 |

**Observations:**
- IWAE k=50 achieves a lower (tighter) training loss than k=5, even with fewer epochs — confirming the paper's theoretical result that the bound improves with more importance samples.
- Higher k leads to lower KL divergence (9.37 vs 12.23), indicating richer use of the latent space, consistent with the paper's finding that IWAEs learn more active latent dimensions.
- Note: IWAE loss values are not directly comparable to Beta-VAE values since IWAE uses the full log-probability (including normalization constants) while Beta-VAE uses MSE + KL.
- Generated samples are visually similar in quality to Beta-VAE, which is expected since both share the same decoder architecture.

WandB: [Beta-VAE & IWAE experiments](https://wandb.ai/gmode-free-university-of-tbilisi-/cifar10-generative-models)

---

wandb report link for Beta-VAE and IWAE: https://wandb.ai/gmode-free-university-of-tbilisi-/cifar10-generative-models/reports/Generative-Models-on-CIFAR-10-Beta-VAE-and-IWAE--VmlldzoxNTc5NzYzMA?accessToken=dzig5jj0487jndn2vbgw8bydpftu01tysb70357raf3w6krfj8b629gwr5j7os7m

## References

- Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
- Higgins, I., et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR.
- Burda, Y., Grosse, R., & Salakhutdinov, R. (2016). *Importance Weighted Autoencoders.* ICLR. [arXiv:1509.00519](https://arxiv.org/abs/1509.00519)
