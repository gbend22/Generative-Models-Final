"""
Importance Weighted Autoencoder (IWAE)
Burda, Grosse & Salakhutdinov (2016)

Uses the same encoder/decoder architecture as Beta-VAE but trains
with the IWAE objective using k importance-weighted samples.
"""
import torch
import torch.nn as nn
from src.models.beta_vae import Encoder, Decoder


class IWAE(nn.Module):
    """
    Importance Weighted Autoencoder.

    The architecture is identical to VAE - the difference is only in the
    training objective (loss function).
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def reparameterize(self, mu, log_var):
        """
        Standard reparameterization trick: z = mu + std * eps

        Args:
            mu: [B, latent_dim]
            log_var: [B, latent_dim]
        Returns:
            z: [B, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparameterize_k(self, mu, log_var, k):
        """
        Sample k latent vectors per data point using reparameterization.

        Args:
            mu: [B, latent_dim]
            log_var: [B, latent_dim]
            k: Number of importance samples
        Returns:
            z: [B, k, latent_dim]
        """
        B, D = mu.shape
        std = torch.exp(0.5 * log_var)
        # Expand for k samples: [B, D] -> [B, 1, D] -> [B, k, D]
        mu_expanded = mu.unsqueeze(1).expand(B, k, D)
        std_expanded = std.unsqueeze(1).expand(B, k, D)
        eps = torch.randn_like(mu_expanded)  # [B, k, D]
        z = mu_expanded + eps * std_expanded  # [B, k, D]
        return z

    def forward(self, x, k=1):
        """
        Forward pass for IWAE.

        Args:
            x: Input images [B, C, H, W]
            k: Number of importance samples
        Returns:
            x_recon: Reconstructed images [B*k, C, H, W]
            mu: Mean of q(z|x) [B, latent_dim]
            log_var: Log variance of q(z|x) [B, latent_dim]
            z: Sampled latent vectors [B, k, latent_dim]
        """
        # Encode once
        mu, log_var = self.encoder(x)  # [B, D]

        # Sample k latent vectors
        z = self.reparameterize_k(mu, log_var, k)  # [B, k, D]

        # Decode all k samples
        B, K, D = z.shape
        z_flat = z.reshape(B * K, D)  # [B*k, D]
        x_recon = self.decoder(z_flat)  # [B*k, C, H, W]

        return x_recon, mu, log_var, z

    def sample(self, n_samples, device):
        """
        Generate samples from prior p(z) = N(0, I).

        Args:
            n_samples: Number of images to generate
            device: Device to use
        Returns:
            Generated images [n_samples, C, H, W]
        """
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z)

    def reconstruct(self, x):
        """
        Reconstruct images using a single sample (for visualization).

        Args:
            x: Input images [B, C, H, W]
        Returns:
            Reconstructed images [B, C, H, W]
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z)
