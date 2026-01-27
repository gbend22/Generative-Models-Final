"""
Denoising Score Matching Loss for NCSN.

Implements the training objective from Song & Ermon (2019):

    L(θ) = (1/L) Σ_{i=1}^{L} λ(σ_i) * E_{p(x)} E_{x̃ ~ N(x, σ_i²I)}
           [ ||s_θ(x̃, σ_i) - ∇_{x̃} log q_σ(x̃|x)||² ]

where:
- s_θ(x̃, σ) is the score network
- ∇_{x̃} log q_σ(x̃|x) = -(x̃ - x) / σ² is the target score
- λ(σ) = σ² is the weighting function that balances different noise levels

References:
- Song & Ermon (2019): "Generative Modeling by Estimating Gradients of the Data Distribution"
- Vincent (2011): "A Connection Between Score Matching and Denoising Autoencoders"
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def dsm_loss(
        score_net: nn.Module,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the Denoising Score Matching loss.

    This is the core training objective for NCSN. For each sample:
    1. Sample a noise level σ_i uniformly from {σ_1, ..., σ_L}
    2. Add Gaussian noise: x̃ = x + σ_i * ε, where ε ~ N(0, I)
    3. Compute target score: target = -ε / σ_i = -(x̃ - x) / σ_i²
    4. Predict score with network: pred = s_θ(x̃, i)
    5. Compute weighted MSE: loss = σ_i² * ||pred - target||²

    The σ² weighting ensures each noise level contributes equally to the loss.

    Args:
        score_net: The score network s_θ(x, σ)
        x: Clean images of shape (B, C, H, W)
        sigmas: Tensor of noise levels of shape (L,)
        labels: Optional pre-selected noise level indices (B,)

    Returns:
        loss: Scalar loss value
        info: Dict with additional information for logging
    """
    # Sample noise level indices uniformly
    if labels is None:
        labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)

    # Get sigma values for each sample
    used_sigmas = sigmas[labels]  # (B,)
    used_sigmas = used_sigmas.view(-1, 1, 1, 1)  # (B, 1, 1, 1) for broadcasting

    # Sample noise
    noise = torch.randn_like(x)  # ε ~ N(0, I)

    # Perturb data: x̃ = x + σ * ε
    perturbed_x = x + used_sigmas * noise

    # Compute target score: -(x̃ - x) / σ² = -ε / σ
    # Note: We actually compute -ε/σ which equals the true score ∇ log q_σ(x̃|x)
    target_score = -noise / used_sigmas

    # Get score network prediction
    predicted_score = score_net(perturbed_x, labels)

    # Compute per-sample squared error
    # ||s_θ(x̃, σ) - target||²
    squared_error = (predicted_score - target_score).pow(2)
    squared_error = squared_error.view(x.shape[0], -1).sum(dim=1)  # Sum over pixels

    # Weight by σ² to balance across noise levels
    # This ensures each noise level contributes approximately equally
    weighted_error = squared_error * used_sigmas.squeeze().pow(2)

    # Average over batch
    loss = weighted_error.mean()

    # Collect info for logging
    info = {
        'loss': loss.item(),
        'mse_unweighted': squared_error.mean().item(),
        'sigma_mean': used_sigmas.mean().item(),
        'score_norm': predicted_score.norm(dim=(1, 2, 3)).mean().item(),
    }

    return loss, info


class DSMLoss(nn.Module):
    """
    Denoising Score Matching Loss as a Module.

    Convenient wrapper around dsm_loss function.

    Args:
        sigmas: Tensor of noise levels (L,)
        anneal_power: Power for noise level annealing (default: 2.0 for σ²)
    """

    def __init__(self, sigmas: torch.Tensor, anneal_power: float = 2.0):
        super().__init__()
        self.register_buffer('sigmas', sigmas)
        self.anneal_power = anneal_power

    def forward(
            self,
            score_net: nn.Module,
            x: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss.

        Args:
            score_net: Score network
            x: Clean images (B, C, H, W)
            labels: Optional noise level indices (B,)

        Returns:
            loss: Scalar loss
            info: Logging dict
        """
        return dsm_loss(score_net, x, self.sigmas, labels)


def dsm_loss_conditional(
        score_net: nn.Module,
        x: torch.Tensor,
        sigma: float,
        sigma_idx: int,
) -> torch.Tensor:
    """
    DSM loss for a specific noise level.

    Useful for analysis and debugging - computes loss at a single σ.

    Args:
        score_net: Score network
        x: Clean images (B, C, H, W)
        sigma: Noise level value
        sigma_idx: Index of the noise level

    Returns:
        Loss value
    """
    B = x.shape[0]
    device = x.device

    # Create label tensor
    labels = torch.full((B,), sigma_idx, dtype=torch.long, device=device)

    # Add noise
    noise = torch.randn_like(x)
    perturbed_x = x + sigma * noise

    # Target and prediction
    target = -noise / sigma
    predicted = score_net(perturbed_x, labels)

    # Weighted loss
    loss = (sigma ** 2) * ((predicted - target) ** 2).sum(dim=(1, 2, 3)).mean()

    return loss


# For analysis: loss per noise level
def compute_loss_per_sigma(
        score_net: nn.Module,
        x: torch.Tensor,
        sigmas: torch.Tensor,
) -> dict:
    """
    Compute DSM loss for each noise level separately.

    Useful for analyzing model behavior across noise scales.

    Args:
        score_net: Score network
        x: Clean images (B, C, H, W)
        sigmas: All noise levels (L,)

    Returns:
        Dict mapping sigma index to loss value
    """
    losses = {}
    score_net.eval()

    with torch.no_grad():
        for i, sigma in enumerate(sigmas):
            loss = dsm_loss_conditional(score_net, x, sigma.item(), i)
            losses[i] = {
                'sigma': sigma.item(),
                'loss': loss.item(),
            }

    return losses