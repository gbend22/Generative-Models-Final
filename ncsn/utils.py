"""
Utility functions for NCSN.

Includes:
- Noise schedule computation (geometric sequence of σ values)
- Exponential Moving Average (EMA) for model weights
- Checkpoint saving/loading
- Image utilities

References:
- Song & Ermon (2019): Noise schedule design
- Song & Ermon (2020): EMA and improved noise schedule
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import os
from typing import Optional, Dict, Any
import yaml


def get_sigmas(
        sigma_begin: float,
        sigma_end: float,
        num_classes: int,
) -> torch.Tensor:
    """
    Compute geometric sequence of noise levels.

    The noise levels form a geometric sequence:
    σ_i = σ_1 * (σ_L / σ_1)^((i-1)/(L-1))

    This ensures log-spaced noise levels, which is important because:
    1. Large σ covers the entire data range
    2. Small σ allows fine-grained details
    3. Geometric spacing gives equal emphasis to each scale

    Args:
        sigma_begin: Largest noise level σ_1
        sigma_end: Smallest noise level σ_L
        num_classes: Number of noise levels L

    Returns:
        Tensor of shape (L,) with noise levels from largest to smallest
    """
    sigmas = torch.tensor(
        np.exp(
            np.linspace(
                np.log(sigma_begin),
                np.log(sigma_end),
                num_classes
            )
        )
    ).float()

    return sigmas


def get_sigmas_from_data(
        data_loader: torch.utils.data.DataLoader,
        num_classes: int,
        percentile_begin: float = 99.0,
        percentile_end: float = 0.01,
) -> torch.Tensor:
    """
    Compute noise levels based on data statistics (NCSNv2 approach).

    σ_1 should be large enough that q_σ1(x) ≈ N(0, σ_1²I)
    σ_L should be small enough that x + σ_L*ε ≈ x

    Args:
        data_loader: DataLoader for the training data
        num_classes: Number of noise levels
        percentile_begin: Percentile for σ_1 (high noise)
        percentile_end: Percentile for σ_L (low noise)

    Returns:
        Tensor of noise levels
    """
    # Compute pairwise distances in a batch
    all_distances = []

    for batch, _ in data_loader:
        batch = batch.view(batch.shape[0], -1)  # Flatten
        # Compute pairwise L2 distances
        dists = torch.cdist(batch, batch, p=2)
        # Get upper triangular (unique pairs)
        triu_indices = torch.triu_indices(dists.shape[0], dists.shape[1], offset=1)
        all_distances.append(dists[triu_indices[0], triu_indices[1]])

        if len(all_distances) > 10:  # Sample from subset
            break

    all_distances = torch.cat(all_distances)

    # σ_1: ~max pairwise distance (cover entire data range)
    sigma_begin = np.percentile(all_distances.numpy(), percentile_begin)

    # σ_L: ~min relevant distance
    sigma_end = np.percentile(all_distances.numpy(), percentile_end)

    return get_sigmas(sigma_begin, sigma_end, num_classes)


class EMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters that is updated as:
    shadow = decay * shadow + (1 - decay) * current

    This provides a smoothed version of the model that often generates
    better samples than the raw trained model.

    Args:
        model: The model to track
        decay: EMA decay rate (typically 0.999 or 0.9999)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        self.register()

    def register(self):
        """Register current model parameters as shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                        self.decay * self.shadow[name] +
                        (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow weights to model (for sampling)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights to model (after sampling)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state for checkpointing."""
        return {
            'shadow': copy.deepcopy(self.shadow),
            'decay': self.decay,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        ema: Optional[EMA],
        epoch: int,
        step: int,
        loss: float,
        config: dict,
        path: str,
):
    """
    Save training checkpoint.

    Args:
        model: The model
        optimizer: Optimizer state
        ema: EMA object (optional)
        epoch: Current epoch
        step: Current step
        loss: Current loss
        config: Training config
        path: Save path
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'config': config,
    }

    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        ema: Optional[EMA] = None,
        device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        ema: EMA to load state into (optional)
        device: Device to load to

    Returns:
        Dict with epoch, step, loss info
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', 0),
        'config': checkpoint.get('config', {}),
    }


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    """Normalize images from [0, 1] to [-1, 1]."""
    return x * 2 - 1


def unnormalize_to_zero_to_one(x: torch.Tensor) -> torch.Tensor:
    """Unnormalize images from [-1, 1] to [0, 1]."""
    return (x + 1) / 2


def make_grid(
        images: torch.Tensor,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
) -> torch.Tensor:
    """
    Make a grid of images for visualization.

    Args:
        images: (B, C, H, W) tensor
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize to [0, 1]

    Returns:
        Grid image tensor (C, H', W')
    """
    from torchvision.utils import make_grid as tv_make_grid

    if normalize:
        images = images.clamp(-1, 1)
        images = unnormalize_to_zero_to_one(images)

    return tv_make_grid(images, nrow=nrow, padding=padding)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)