"""
NCSN (Noise Conditional Score Network) Architecture
Implements RefineNet-based score network for CIFAR-10

Based on: "Generative Modeling by Estimating Gradients of the Data Distribution"
Song & Ermon, NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.refineNet import RefineNet

class NCSN(nn.Module):
    """
    Noise Conditional Score Network

    Learns score function ∇_x log p(x|σ) at multiple noise levels
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Score network
        self.score_net = RefineNet(
            in_channels=config.get('in_channels', 3),
            base_channels=config.get('base_channels', 128),
            num_classes=config.get('num_noise_levels', 10)
        )

        # Noise levels (geometric sequence)
        self.register_buffer(
            'sigmas',
            self._get_sigmas(
                sigma_begin=config.get('sigma_begin', 1.0),
                sigma_end=config.get('sigma_end', 0.01),
                num_classes=config.get('num_noise_levels', 10)
            )
        )

    def _get_sigmas(self, sigma_begin, sigma_end, num_classes):
        """Generate geometric sequence of noise levels"""
        return torch.exp(torch.linspace(
            np.log(sigma_begin),
            np.log(sigma_end),
            num_classes
        ))

    def forward(self, x, labels=None):
        """
        Compute score estimates

        Args:
            x: Input images [B, C, H, W]
            labels: Noise level indices [B] (if None, use random)

        Returns:
            scores: Score estimates [B, C, H, W]
            labels: Used noise levels [B]
        """
        if labels is None:
            # Sample random noise levels during training
            labels = torch.randint(
                0, len(self.sigmas), (x.shape[0],), device=x.device
            )

        # Get scores conditioned on noise level
        scores = self.score_net(x, labels)

        # Scale by noise level (important for score matching)
        used_sigmas = self.sigmas[labels].view(-1, 1, 1, 1)
        scores = scores / used_sigmas

        return scores, labels

    @torch.no_grad()
    def sample(self, batch_size, device, n_steps_each=100, step_lr=0.00002):
        """
        Generate samples using Annealed Langevin Dynamics

        Args:
            batch_size: Number of samples
            device: Device to generate on
            n_steps_each: Langevin steps per noise level
            step_lr: Step size for Langevin dynamics

        Returns:
            Generated samples [B, 3, 32, 32]
        """
        # Start from pure noise
        x = torch.randn(batch_size, 3, 32, 32, device=device)

        # Anneal from high to low noise
        for sigma_idx in range(len(self.sigmas)):
            sigma = self.sigmas[sigma_idx]
            labels = torch.ones(batch_size, device=device, dtype=torch.long) * sigma_idx

            # Adjusted step size for this noise level
            step_size = step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(n_steps_each):
                # Add noise
                noise = torch.randn_like(x)

                # Get score estimate
                scores, _ = self.forward(x, labels)

                # Langevin dynamics update
                x = x + step_size * scores + torch.sqrt(2 * step_size) * noise

                # Clamp to valid range
                x = torch.clamp(x, -1, 1)

        return x


# Test the model
if __name__ == "__main__":
    # Configuration
    config = {
        'in_channels': 3,
        'base_channels': 128,
        'num_noise_levels': 10,
        'sigma_begin': 1.0,
        'sigma_end': 0.01
    }

    # Create model
    model = NCSN(config)

    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    scores, labels = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Score shape: {scores.shape}")
    print(f"Labels: {labels}")
    print(f"Noise levels: {model.sigmas}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test sampling
    print("\nTesting sampling...")
    samples = model.sample(batch_size=2, device='cpu', n_steps_each=10)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.2f}, {samples.max():.2f}]")