"""
Annealed Langevin Dynamics Sampler
Based on Song & Ermon (2019)

The sampler progressively denoises samples by running Langevin dynamics
at each noise level, from highest to lowest.
"""

import torch
import numpy as np
from tqdm import tqdm


class AnnealedLangevinDynamics:
    """
    Annealed Langevin Dynamics sampler

    Algorithm:
    1. Initialize x from prior (e.g., Gaussian noise)
    2. For each noise level σ_i (from large to small):
        - Run T steps of Langevin dynamics:
          x_{t+1} = x_t + ε_i * s_θ(x_t, σ_i) + sqrt(2*ε_i) * z_t
        where ε_i is the step size and z_t ~ N(0, I)
    3. Return final sample
    """

    def __init__(self, config, sigmas):
        """
        Args:
            config: experiment configuration
            sigmas: array of noise levels [σ_1, ..., σ_L]
        """
        self.config = config
        self.sigmas = sigmas
        self.n_steps_each = config['sampling']['n_steps_each']
        self.step_lr = config['sampling']['step_lr']
        self.denoise = config['sampling']['denoise']

    @torch.no_grad()
    def sample(self, model, batch_size, device, return_intermediate=False, verbose=True):
        """
        Generate samples using annealed Langevin dynamics

        Args:
            model: trained NCSN model
            batch_size: number of samples to generate
            device: torch device
            return_intermediate: if True, return samples at each noise level
            verbose: show progress bar

        Returns:
            samples: (B, C, H, W) or list of samples at each level
        """
        model.eval()

        # Initialize from Gaussian noise
        x = torch.randn(
            batch_size,
            self.config['num_channels'],
            self.config['image_size'],
            self.config['image_size'],
            device=device
        )

        intermediate_samples = []

        # Iterate through noise levels (from high to low)
        iterator = enumerate(self.sigmas)
        if verbose:
            iterator = tqdm(iterator, total=len(self.sigmas), desc='Sampling')

        for i, sigma in iterator:
            # Step size: ε_i = α * σ_i²
            step_size = self.step_lr * (sigma ** 2)

            # Create labels for this noise level
            labels = torch.ones(batch_size, dtype=torch.long, device=device) * i

            # Run Langevin dynamics for n_steps_each iterations
            for step in range(self.n_steps_each):
                # Sample noise z ~ N(0, I)
                noise = torch.randn_like(x)

                # Compute score s_θ(x, σ_i)
                scores = model(x, labels)

                # Langevin update: x ← x + ε*s_θ(x,σ) + sqrt(2ε)*z
                x = x + step_size * scores + torch.sqrt(2 * step_size) * noise

            if return_intermediate:
                intermediate_samples.append(x.clone().cpu())

        # Optional: final denoising step
        # Use one more step with the smallest noise level
        if self.denoise:
            labels = torch.ones(batch_size, dtype=torch.long, device=device) * (len(self.sigmas) - 1)
            scores = model(x, labels)
            x = x + (self.sigmas[-1] ** 2) * scores

        # Clamp to valid range
        x = torch.clamp(x, -1.0, 1.0)

        if return_intermediate:
            return x, intermediate_samples
        else:
            return x

    @torch.no_grad()
    def sample_progressive(self, model, batch_size, device):
        """
        Sample and return progression through all noise levels
        Useful for visualization

        Returns:
            list of tensors, one for each noise level
        """
        return self.sample(model, batch_size, device, return_intermediate=True, verbose=True)


class LangevinDynamics:
    """
    Standard Langevin Dynamics (single noise level)
    Mainly for testing/debugging
    """

    def __init__(self, step_size=0.00002, n_steps=1000):
        self.step_size = step_size
        self.n_steps = n_steps

    @torch.no_grad()
    def sample(self, model, sigma_idx, batch_size, image_size, channels, device):
        """
        Sample using Langevin dynamics at a single noise level

        Args:
            model: score network
            sigma_idx: index of noise level to use
            batch_size: number of samples
            image_size: size of images
            channels: number of channels
            device: torch device
        """
        model.eval()

        # Initialize
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)
        labels = torch.ones(batch_size, dtype=torch.long, device=device) * sigma_idx

        for step in tqdm(range(self.n_steps), desc=f'Langevin σ={sigma_idx}'):
            noise = torch.randn_like(x)
            scores = model(x, labels)
            x = x + self.step_size * scores + torch.sqrt(2 * torch.tensor(
    self.step_size,
    device=device,
    dtype=x.dtype
)) * noise

        return torch.clamp(x, -1.0, 1.0)


def get_sampler(config, sigmas):
    """Factory function to create sampler"""
    return AnnealedLangevinDynamics(config, sigmas)


if __name__ == '__main__':
    # Test sampler
    from src.models.NCSN import NCSN
    from configs.ncsn_cifar10 import config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model = NCSN(config).to(device)
    model.eval()

    # Create sampler
    sampler = get_sampler(config, model.sigmas.cpu().numpy())

    print("Testing sampler...")
    samples = sampler.sample(model, batch_size=4, device=device, verbose=True)

    print(f"\nGenerated samples shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")

    # Test progressive sampling
    print("\nTesting progressive sampling...")
    final_samples, intermediate = sampler.sample_progressive(model, batch_size=2, device=device)
    print(f"Number of intermediate samples: {len(intermediate)}")