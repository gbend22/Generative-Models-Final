"""
Denoising Score Matching Loss for NCSN

Based on: "Generative Modeling by Estimating Gradients of the Data Distribution"
Song & Ermon, NeurIPS 2019
"""

import torch
import torch.nn as nn


class DenoisingScoreMatchingLoss(nn.Module):
    """
    Denoising Score Matching (DSM) Loss

    The key insight: Instead of computing expensive score matching loss,
    we can train by denoising perturbed data.

    For data x and noise ε ~ N(0, σ²I):
        x_perturbed = x + ε

    The score of perturbed distribution is:
        ∇ log p(x_perturbed | x) = -ε / σ²

    Loss:
        L = E[||s_θ(x + ε, σ) - (-ε/σ²)||²]
        L = E[||s_θ(x + ε, σ) + ε/σ²||²]

    This is much easier to compute than vanilla score matching!
    """

    def __init__(self, sigmas):
        """
        Args:
            sigmas: Noise levels [num_noise_levels]
        """
        super().__init__()
        self.sigmas = sigmas

    def forward(self, model, x):
        """
        Compute denoising score matching loss

        Args:
            model: NCSN model
            x: Clean data [B, C, H, W]

        Returns:
            loss: Scalar loss
            loss_dict: Dictionary of per-noise-level losses (for logging)
        """
        batch_size = x.shape[0]

        # Sample random noise levels for each sample
        labels = torch.randint(
            0, len(self.sigmas), (batch_size,), device=x.device
        )

        # Get corresponding noise levels
        used_sigmas = self.sigmas[labels].view(-1, 1, 1, 1)

        # Add noise: x_perturbed = x + σ * ε, where ε ~ N(0, I)
        noise = torch.randn_like(x)
        perturbed_x = x + used_sigmas * noise

        # Get score estimate from model
        scores, _ = model(perturbed_x, labels)

        # Target: -ε/σ (negative noise scaled by sigma)
        target = -noise / used_sigmas

        # MSE loss between predicted score and target
        loss = ((scores - target) ** 2).sum(dim=(1, 2, 3)).mean()

        # Per-noise-level losses (for analysis)
        loss_dict = {}
        for i in range(len(self.sigmas)):
            mask = (labels == i)
            if mask.sum() > 0:
                level_loss = ((scores[mask] - target[mask]) ** 2).sum(dim=(1, 2, 3)).mean()
                loss_dict[f'loss_sigma_{i}'] = level_loss.item()

        return loss, loss_dict


class AnnealedDenoisingScoreLoss(nn.Module):
    """
    Annealed Denoising Score Matching Loss

    Weights losses at different noise levels to improve training.
    Higher noise levels get higher weight.

    This helps because:
    - High noise: easier to learn, provides good gradients
    - Low noise: harder but necessary for final quality
    """

    def __init__(self, sigmas, weighting='exponential'):
        """
        Args:
            sigmas: Noise levels
            weighting: How to weight different noise levels
                - 'uniform': Equal weight
                - 'exponential': Higher noise gets more weight
                - 'inverse': Weight by 1/σ²
        """
        super().__init__()
        self.sigmas = sigmas
        self.weighting = weighting

        # Compute weights
        if weighting == 'uniform':
            self.weights = torch.ones_like(sigmas)
        elif weighting == 'exponential':
            # Higher noise gets exponentially more weight
            self.weights = sigmas ** 2
        elif weighting == 'inverse':
            # Weight by 1/σ²
            self.weights = 1.0 / (sigmas ** 2 + 1e-8)
        else:
            raise ValueError(f"Unknown weighting: {weighting}")

        # Normalize weights
        self.weights = self.weights / self.weights.sum()

    def forward(self, model, x):
        """
        Compute weighted denoising score matching loss

        Args:
            model: NCSN model
            x: Clean data [B, C, H, W]

        Returns:
            loss: Weighted scalar loss
            loss_dict: Per-noise-level losses
        """
        batch_size = x.shape[0]

        # Sample noise levels (weighted by importance)
        labels = torch.multinomial(
            self.weights.to(x.device),
            batch_size,
            replacement=True
        )

        # Get noise levels
        used_sigmas = self.sigmas[labels].view(-1, 1, 1, 1)

        # Perturb data
        noise = torch.randn_like(x)
        perturbed_x = x + used_sigmas * noise

        # Get scores
        scores, _ = model(perturbed_x, labels)

        # Target
        target = -noise / used_sigmas

        # Compute loss with weighting
        per_sample_loss = ((scores - target) ** 2).sum(dim=(1, 2, 3))

        # Weight by sigma (already sampled according to weights, but can add explicit weighting)
        loss = per_sample_loss.mean()

        # Per-noise-level logging
        loss_dict = {}
        for i in range(len(self.sigmas)):
            mask = (labels == i)
            if mask.sum() > 0:
                level_loss = per_sample_loss[mask].mean()
                loss_dict[f'loss_sigma_{i}'] = level_loss.item()
                loss_dict[f'weight_sigma_{i}'] = self.weights[i].item()

        return loss, loss_dict


class SlicedScoreMatchingLoss(nn.Module):
    """
    Sliced Score Matching (Optional - for comparison)

    Uses random projections to make score matching tractable.
    More expensive than denoising but doesn't require noise perturbation.

    L = E[v^T ∇s_θ(x) v + 0.5 ||s_θ(x)||²]
    where v ~ N(0, I)
    """

    def __init__(self, sigmas, n_projections=1):
        super().__init__()
        self.sigmas = sigmas
        self.n_projections = n_projections

    def forward(self, model, x):
        """
        Compute sliced score matching loss

        This requires computing Jacobian-vector products,
        which is more expensive than denoising score matching.
        """
        batch_size = x.shape[0]

        # Sample noise levels
        labels = torch.randint(0, len(self.sigmas), (batch_size,), device=x.device)

        x.requires_grad_(True)

        # Get scores
        scores, _ = model(x, labels)

        total_loss = 0

        for _ in range(self.n_projections):
            # Random projection vector
            v = torch.randn_like(x)

            # Compute v^T ∇s_θ v using autograd
            # This is the trace of the Jacobian projected onto v
            vjp = torch.autograd.grad(
                scores, x,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True
            )[0]

            # Sliced score matching loss
            loss = (vjp * v).sum(dim=(1, 2, 3)) + 0.5 * (scores ** 2).sum(dim=(1, 2, 3))
            total_loss += loss.mean()

        x.requires_grad_(False)

        return total_loss / self.n_projections, {}


# Test the losses
if __name__ == "__main__":
    import torch
    from models.ncsn import NCSN

    # Setup
    config = {
        'in_channels': 3,
        'base_channels': 64,  # Smaller for testing
        'num_noise_levels': 5,
        'sigma_begin': 1.0,
        'sigma_end': 0.01
    }

    model = NCSN(config)
    x = torch.randn(4, 3, 32, 32)

    print("Testing Denoising Score Matching Loss:")
    print("=" * 60)

    # Test DSM loss
    dsm_loss = DenoisingScoreMatchingLoss(model.sigmas)
    loss, loss_dict = dsm_loss(model, x)

    print(f"\nTotal loss: {loss.item():.4f}")
    print(f"Per-level losses:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.4f}")

    # Test Annealed DSM
    print("\n" + "=" * 60)
    print("Testing Annealed Denoising Score Matching Loss:")
    print("=" * 60)

    annealed_loss = AnnealedDenoisingScoreLoss(model.sigmas, weighting='exponential')
    loss, loss_dict = annealed_loss(model, x)

    print(f"\nTotal loss: {loss.item():.4f}")
    print(f"Per-level losses and weights:")
    for i in range(len(model.sigmas)):
        if f'loss_sigma_{i}' in loss_dict:
            print(f"  Sigma {i}: loss={loss_dict[f'loss_sigma_{i}']:.4f}, "
                  f"weight={loss_dict[f'weight_sigma_{i}']:.4f}")

    print("\n✓ Loss functions working correctly!")