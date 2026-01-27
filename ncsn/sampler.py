"""
Annealed Langevin Dynamics Sampler for NCSN.

Implements the sampling procedure from Song & Ermon (2019):

For i = 1 to L (noise levels from large to small):
    α_i = ε * σ_i² / σ_L²  (step size)
    For t = 1 to T (Langevin steps):
        z ~ N(0, I)
        x ← x + (α_i / 2) * s_θ(x, σ_i) + √α_i * z

The key insight is to start from high noise (where the score is well-defined
everywhere) and gradually anneal to low noise, following the score field
to reach the data manifold.

References:
- Song & Ermon (2019): "Generative Modeling by Estimating Gradients of the Data Distribution"
- Song & Ermon (2020): "Improved Techniques for Training Score-Based Generative Models"
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, List
from tqdm import tqdm


class AnnealedLangevinDynamics:
    """
    Annealed Langevin Dynamics sampler.

    Generates samples by starting from pure noise and iteratively refining
    using the learned score function, with gradually decreasing noise levels.

    Args:
        score_net: Trained score network s_θ(x, σ)
        sigmas: Tensor of noise levels (L,), from largest to smallest
        num_steps: Number of Langevin steps T per noise level
        step_lr: Base step size ε
        denoise: Whether to apply denoising at the final step
        device: Device to run sampling on
    """

    def __init__(
            self,
            score_net: nn.Module,
            sigmas: torch.Tensor,
            num_steps: int = 100,
            step_lr: float = 2e-5,
            denoise: bool = True,
            device: str = 'cuda',
    ):
        self.score_net = score_net
        self.sigmas = sigmas.to(device)
        self.num_steps = num_steps
        self.step_lr = step_lr
        self.denoise = denoise
        self.device = device

        self.score_net.eval()

    @torch.no_grad()
    def sample(
            self,
            batch_size: int,
            img_shape: tuple = (3, 32, 32),
            return_intermediate: bool = False,
            verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using annealed Langevin dynamics.

        Args:
            batch_size: Number of samples to generate
            img_shape: Shape of each image (C, H, W)
            return_intermediate: If True, return samples at each noise level
            verbose: If True, show progress bar

        Returns:
            Generated samples of shape (B, C, H, W)
            If return_intermediate, returns list of tensors
        """
        # Initialize from Gaussian noise
        # x_0 ~ N(0, σ_1² I) - start with noise at the largest scale
        x = torch.randn(batch_size, *img_shape, device=self.device)
        x = x * self.sigmas[0]  # Scale by largest sigma

        intermediates = [x.clone()] if return_intermediate else None

        # Iterate through noise levels (from large to small)
        iterator = enumerate(self.sigmas)
        if verbose:
            iterator = tqdm(list(iterator), desc="Sampling")

        for i, sigma in iterator:
            # Create label tensor for this noise level
            labels = torch.full((batch_size,), i, dtype=torch.long, device=self.device)

            # Compute step size: α_i = ε * σ_i² / σ_L²
            # This scales the step size proportionally to the noise level
            step_size = self.step_lr * (sigma / self.sigmas[-1]) ** 2

            # Langevin dynamics at this noise level
            for t in range(self.num_steps):
                # Get score
                score = self.score_net(x, labels)

                # Sample noise for Langevin step
                z = torch.randn_like(x)

                # Langevin update: x ← x + (α/2) * s_θ(x, σ) + √α * z
                x = x + (step_size / 2) * score + torch.sqrt(step_size) * z

            if return_intermediate:
                intermediates.append(x.clone())

        # Optional denoising step at the end
        # This uses the score at the smallest noise level to do one final update
        if self.denoise:
            # One more score evaluation at smallest noise level
            labels = torch.full((batch_size,), len(self.sigmas) - 1,
                                dtype=torch.long, device=self.device)
            score = self.score_net(x, labels)

            # Denoising: x ← x + σ_L² * s_θ(x, σ_L)
            # This is essentially a Tweedie denoising step
            x = x + self.sigmas[-1] ** 2 * score

        if return_intermediate:
            return intermediates
        return x

    @torch.no_grad()
    def sample_progressive(
            self,
            batch_size: int,
            img_shape: tuple = (3, 32, 32),
            save_freq: int = 1,
    ) -> List[torch.Tensor]:
        """
        Generate samples and save intermediate results.

        Useful for visualization of the sampling process.

        Args:
            batch_size: Number of samples
            img_shape: Image shape (C, H, W)
            save_freq: Save intermediate every N noise levels

        Returns:
            List of intermediate samples
        """
        x = torch.randn(batch_size, *img_shape, device=self.device)
        x = x * self.sigmas[0]

        results = []

        for i, sigma in enumerate(tqdm(self.sigmas, desc="Progressive sampling")):
            labels = torch.full((batch_size,), i, dtype=torch.long, device=self.device)
            step_size = self.step_lr * (sigma / self.sigmas[-1]) ** 2

            for t in range(self.num_steps):
                score = self.score_net(x, labels)
                z = torch.randn_like(x)
                x = x + (step_size / 2) * score + torch.sqrt(step_size) * z

            if i % save_freq == 0:
                results.append(x.clone())

        # Final denoised result
        if self.denoise:
            labels = torch.full((batch_size,), len(self.sigmas) - 1,
                                dtype=torch.long, device=self.device)
            score = self.score_net(x, labels)
            x = x + self.sigmas[-1] ** 2 * score

        results.append(x.clone())
        return results


def sample_from_model(
        score_net: nn.Module,
        sigmas: torch.Tensor,
        num_samples: int = 64,
        num_steps: int = 100,
        step_lr: float = 2e-5,
        denoise: bool = True,
        device: str = 'cuda',
        verbose: bool = True,
) -> torch.Tensor:
    """
    Convenience function for sampling.

    Args:
        score_net: Trained score network
        sigmas: Noise levels tensor
        num_samples: Number of samples to generate
        num_steps: Langevin steps per noise level
        step_lr: Base step size
        denoise: Apply final denoising
        device: Device for sampling
        verbose: Show progress

    Returns:
        Generated samples (B, C, H, W)
    """
    sampler = AnnealedLangevinDynamics(
        score_net=score_net,
        sigmas=sigmas,
        num_steps=num_steps,
        step_lr=step_lr,
        denoise=denoise,
        device=device,
    )

    return sampler.sample(
        batch_size=num_samples,
        img_shape=(3, 32, 32),
        verbose=verbose,
    )


# Alternative sampler with different step size schedule
class AnnealedLangevinDynamicsV2:
    """
    Improved Annealed Langevin Dynamics (from NCSNv2).

    Uses a different step size schedule and supports EMA models.

    Args:
        score_net: Score network
        sigmas: Noise levels
        n_steps_each: Steps per noise level
        step_lr: Base learning rate
        denoise: Final denoising
        final_only: Only use final noise level
    """

    def __init__(
            self,
            score_net: nn.Module,
            sigmas: torch.Tensor,
            n_steps_each: int = 100,
            step_lr: float = 0.00002,
            denoise: bool = True,
            final_only: bool = False,
            device: str = 'cuda',
    ):
        self.score_net = score_net
        self.sigmas = sigmas.to(device)
        self.n_steps_each = n_steps_each
        self.step_lr = step_lr
        self.denoise = denoise
        self.final_only = final_only
        self.device = device

    @torch.no_grad()
    def sample(
            self,
            batch_size: int,
            img_shape: tuple = (3, 32, 32),
    ) -> torch.Tensor:
        """Generate samples."""
        self.score_net.eval()

        # Initialize
        x = torch.randn(batch_size, *img_shape, device=self.device)

        with torch.no_grad():
            for c, sigma in enumerate(tqdm(self.sigmas, desc='Sampling')):
                labels = torch.ones(batch_size, device=self.device, dtype=torch.long) * c

                # Adaptive step size
                step_size = self.step_lr * (sigma / self.sigmas[-1]) ** 2

                for _ in range(self.n_steps_each):
                    grad = self.score_net(x, labels)
                    noise = torch.randn_like(x)

                    # Langevin step
                    x = x + step_size * grad / 2 + noise * torch.sqrt(step_size)

        # Denoising
        if self.denoise:
            last_noise = (len(self.sigmas) - 1) * torch.ones(
                batch_size, device=self.device, dtype=torch.long
            )
            x = x + self.sigmas[-1] ** 2 * self.score_net(x, last_noise)

        return x.clamp(-1, 1)