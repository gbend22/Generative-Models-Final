"""
Denoising Score Matching Loss
Based on Song & Ermon (2019, 2020)

The key insight: instead of computing the intractable score matching loss,
we can train on noisy data and use the closed-form conditional score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DenoisingScoreMatching(nn.Module):
    """
    Denoising Score Matching Loss

    For data x ~ p_data and noise ε ~ N(0, σ²I):
    - Perturbed data: x̃ = x + ε
    - Conditional score: ∇_x̃ log p(x̃|x) = -ε/σ²

    Loss: E_x,ε [λ(σ) ||s_θ(x̃, σ) - ∇_x̃ log p(x̃|x)||²]
        = E_x,ε [λ(σ) ||s_θ(x + ε, σ) + ε/σ²||²]

    where λ(σ) is an importance weight
    """

    def __init__(self, sigmas, loss_type='default'):
        """
        Args:
            sigmas: array of noise levels
            loss_type: 'default' or 'weighted'
        """
        super().__init__()
        self.register_buffer('sigmas', torch.tensor(sigmas).float())
        self.loss_type = loss_type

    def get_loss_weight(self, sigma):
        """
        Compute importance weight λ(σ)

        Options:
        - 'default': λ(σ) = σ² (from Song & Ermon 2019)
        - 'weighted': λ(σ) = 1 (uniform weighting)
        """
        if self.loss_type == 'default':
            return sigma ** 2
        elif self.loss_type == 'weighted':
            return torch.ones_like(sigma)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, model, x):
        """
        Compute denoising score matching loss

        Args:
            model: score network
            x: clean data (B, C, H, W)

        Returns:
            loss: scalar loss value
            loss_dict: dictionary with detailed loss information
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample random noise levels for each sample in batch
        labels = torch.randint(0, len(self.sigmas), (batch_size,), device=device)
        used_sigmas = self.sigmas[labels].view(batch_size, 1, 1, 1)

        # Sample noise ε ~ N(0, σ²I)
        noise = torch.randn_like(x) * used_sigmas

        # Perturbed data x̃ = x + ε
        perturbed_x = x + noise

        # Get score estimate from model
        # Model outputs s_θ(x̃, σ)
        target = -noise / (used_sigmas ** 2)  # True conditional score
        scores = model(perturbed_x, labels)

        # Compute loss with importance weighting
        loss_weight = self.get_loss_weight(used_sigmas)
        losses = loss_weight * ((scores - target) ** 2)

        # Average over all dimensions
        loss = torch.mean(losses)

        # Compute per-noise-level statistics for logging
        loss_dict = {
            'loss': loss.item(),
        }

        # Add per-sigma losses for analysis
        for i, sigma in enumerate(self.sigmas):
            mask = (labels == i)
            if mask.sum() > 0:
                sigma_loss = losses[mask].mean()
                loss_dict[f'loss_sigma_{i}'] = sigma_loss.item()

        return loss, loss_dict


class AnnealedDenoisingScoreMatching(nn.Module):
    """
    Annealed version with curriculum learning
    Gradually increases the number of noise levels during training
    """

    def __init__(self, sigmas, warmup_iters=5000):
        super().__init__()
        self.register_buffer('sigmas', torch.tensor(sigmas).float())
        self.warmup_iters = warmup_iters
        self.current_iter = 0

    def forward(self, model, x):
        """
        Args:
            model: score network
            x: clean data (B, C, H, W)

        Returns:
            loss: scalar loss value
            loss_dict: dictionary with detailed loss information
        """
        batch_size = x.shape[0]
        device = x.device

        # Determine active noise levels (curriculum)
        if self.current_iter < self.warmup_iters:
            # Start with only the largest noise level
            max_level = max(1, int((self.current_iter / self.warmup_iters) * len(self.sigmas)))
            active_sigmas = self.sigmas[:max_level]
        else:
            active_sigmas = self.sigmas

        # Sample random noise levels
        labels = torch.randint(0, len(active_sigmas), (batch_size,), device=device)
        used_sigmas = active_sigmas[labels].view(batch_size, 1, 1, 1)

        # Sample noise and create perturbed data
        noise = torch.randn_like(x) * used_sigmas
        perturbed_x = x + noise

        # Compute scores
        # Need to map to full label space
        full_labels = labels  # They're already in [0, max_level)
        if len(active_sigmas) < len(self.sigmas):
            # Map to corresponding indices in full sigma range
            full_labels = (labels * len(self.sigmas) // len(active_sigmas)).long()

        target = -noise / (used_sigmas ** 2)
        scores = model(perturbed_x, full_labels)

        # Compute weighted loss
        loss_weight = used_sigmas ** 2
        losses = loss_weight * ((scores - target) ** 2)
        loss = torch.mean(losses)

        loss_dict = {
            'loss': loss.item(),
            'active_levels': len(active_sigmas),
        }

        self.current_iter += 1

        return loss, loss_dict


def get_loss_fn(config):
    """
    Factory function to create loss function
    """
    sigmas = np.exp(np.linspace(
        np.log(config['noise']['sigma_begin']),
        np.log(config['noise']['sigma_end']),
        config['noise']['num_classes']
    ))

    loss_fn = DenoisingScoreMatching(sigmas, loss_type='default')

    return loss_fn


if __name__ == '__main__':
    # Test loss function
    from src.models.NCSN import NCSN
    from configs.ncsn_cifar10 import config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model and loss
    model = NCSN(config).to(device)
    loss_fn = get_loss_fn(config).to(device)

    # Test data
    batch_size = 8
    x = torch.randn(batch_size, 3, 32, 32).to(device)

    # Compute loss
    loss, loss_dict = loss_fn(model, x)

    print(f"Loss: {loss.item():.4f}")
    print(f"\nDetailed losses:")
    for key, value in loss_dict.items():
        if key != 'loss':
            print(f"  {key}: {value:.4f}")

    # Test backward pass
    loss.backward()
    print(f"\nGradient computed successfully")
    print(f"First layer gradient norm: {model.score_net.conv_first.weight.grad.norm().item():.4f}")