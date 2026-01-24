"""
Noise Conditional Score Network (NCSN)
Based on "Generative Modeling by Estimating Gradients of the Data Distribution"
Song & Ermon, NeurIPS 2019
"""

import torch
import torch.nn as nn
import numpy as np
from unet import RefineNetUNet


class NCSN(nn.Module):
    """
    Noise Conditional Score Network
    Estimates the score function s_θ(x, σ) ≈ ∇_x log p_σ(x)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build the score network
        self.score_net = RefineNetUNet(
            image_channels=config['num_channels'],
            ngf=config['model']['ngf'],
            num_classes=config['model']['num_classes']
        )

        # Register noise levels as buffer (not trainable)
        sigmas = self.get_sigmas()
        self.register_buffer('sigmas', torch.tensor(sigmas).float())

    def get_sigmas(self):
        """
        Generate geometric sequence of noise levels
        """
        sigma_begin = self.config['noise']['sigma_begin']
        sigma_end = self.config['noise']['sigma_end']
        num_classes = self.config['noise']['num_classes']

        sigmas = np.exp(np.linspace(
            np.log(sigma_begin),
            np.log(sigma_end),
            num_classes
        ))

        return sigmas

    def forward(self, x, labels):
        """
        Forward pass through the score network

        Args:
            x: (B, C, H, W) input images
            labels: (B,) noise level indices

        Returns:
            (B, C, H, W) estimated scores (gradients)
        """
        # Get scores from network
        scores = self.score_net(x, labels)

        # The score network outputs the score scaled by sigma
        # We need to divide by sigma to get the actual score
        # This is based on the parameterization in Song & Ermon (2019)
        used_sigmas = self.sigmas[labels].view(-1, 1, 1, 1)
        scores = scores / used_sigmas

        return scores

    @torch.no_grad()
    def sample(self, batch_size, device, return_all=False):
        """
        Generate samples using Annealed Langevin Dynamics

        Args:
            batch_size: number of samples to generate
            device: torch device
            return_all: if True, return samples from all noise levels

        Returns:
            Generated samples (B, C, H, W)
        """
        # Start from random noise
        x = torch.randn(
            batch_size,
            self.config['num_channels'],
            self.config['image_size'],
            self.config['image_size']
        ).to(device)

        all_samples = []

        # Annealed Langevin dynamics
        for i in range(len(self.sigmas)):
            sigma = self.sigmas[i]
            labels = torch.ones(batch_size, dtype=torch.long, device=device) * i
            step_size = self.config['sampling']['step_lr'] * (sigma ** 2)

            for step in range(self.config['sampling']['n_steps_each']):
                # Compute score
                noise = torch.randn_like(x)
                scores = self.forward(x, labels)

                # Langevin dynamics update
                x = x + step_size * scores + torch.sqrt(2 * step_size) * noise

            if return_all:
                all_samples.append(x.clone())

        # Optional: final denoising step
        if self.config['sampling']['denoise']:
            # Use the score at the smallest noise level for one final step
            labels = torch.ones(batch_size, dtype=torch.long, device=device) * (len(self.sigmas) - 1)
            scores = self.forward(x, labels)
            x = x + (self.sigmas[-1] ** 2) * scores

        if return_all:
            return torch.stack(all_samples, dim=1)  # (B, num_sigmas, C, H, W)
        else:
            return x

    def get_num_parameters(self):
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ExponentialMovingAverage:
    """
    Maintains exponential moving averages of model parameters
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


if __name__ == '__main__':
    # Test NCSN
    from configs.ncsn_cifar10 import config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NCSN(config).to(device)

    print(f"Number of parameters: {model.get_num_parameters():,}")
    print(f"Noise levels (sigmas): {model.sigmas.cpu().numpy()}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)

    scores = model(x, labels)
    print(f"\nInput shape: {x.shape}")
    print(f"Scores shape: {scores.shape}")

    # Test sampling
    print("\nTesting sampling...")
    samples = model.sample(batch_size=2, device=device)
    print(f"Generated samples shape: {samples.shape}")