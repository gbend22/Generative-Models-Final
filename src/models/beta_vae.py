import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder network for Beta-VAE.
    Maps input images to latent distribution parameters (mu, log_var).

    Architecture based on Table 1 of the beta-VAE paper,
    adapted for CIFAR-10 (32x32x3) with increased capacity.
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()

        # Convolutional layers with increased channels
        # Input: 32x32x3
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)   # -> 16x16x64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)           # -> 8x8x128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)          # -> 4x4x256
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)          # -> 2x2x512

        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Fully connected layer
        self.fc = nn.Linear(512 * 2 * 2, 512)

        # Output layers for mu and log_var
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)

    def forward(self, x):
        """
        Args:
            x: Input images [B, C, H, W]
        Returns:
            mu: Mean of latent distribution [B, latent_dim]
            log_var: Log variance of latent distribution [B, latent_dim]
        """
        # Convolutional layers with BatchNorm and LeakyReLU
        h = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.2)
        h = F.leaky_relu(self.bn4(self.conv4(h)), 0.2)

        # Flatten
        h = h.view(h.size(0), -1)

        # FC layer
        h = F.leaky_relu(self.fc(h), 0.2)

        # Output mu and log_var
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        return mu, log_var


class Decoder(nn.Module):
    """
    Decoder network for Beta-VAE.
    Maps latent vectors back to image space.

    Architecture mirrors the encoder (transposed convolutions) with increased capacity.
    """
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()

        # FC layers to unflatten
        self.fc = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 512 * 2 * 2)

        # Transposed convolutional layers (decoder)
        # Input: 2x2x512
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # -> 4x4x256
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # -> 8x8x128
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # -> 16x16x64
        self.deconv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)  # -> 32x32x3

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, z):
        """
        Args:
            z: Latent vectors [B, latent_dim]
        Returns:
            x_recon: Reconstructed images [B, C, H, W]
        """
        # FC layers
        h = F.leaky_relu(self.fc(z), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)

        # Reshape to 2D feature maps
        h = h.view(h.size(0), 512, 2, 2)

        # Transposed convolutions with BatchNorm and ReLU
        h = F.relu(self.bn1(self.deconv1(h)))
        h = F.relu(self.bn2(self.deconv2(h)))
        h = F.relu(self.bn3(self.deconv3(h)))

        # Final layer with sigmoid to get values in [0, 1]
        x_recon = torch.sigmoid(self.deconv4(h))

        return x_recon


class BetaVAE(nn.Module):
    """
    Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
    (Higgins et al., ICLR 2017)

    The key modification from standard VAE is the beta hyperparameter that weights
    the KL divergence term:

    L = E[log p(x|z)] - beta * D_KL(q(z|x) || p(z))

    beta > 1 encourages disentangled representations but may sacrifice reconstruction.
    beta = 1 recovers the standard VAE.
    """
    def __init__(self, in_channels=3, latent_dim=32):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + std * epsilon
        This allows gradients to flow through the sampling operation.

        Args:
            mu: Mean of latent distribution [B, latent_dim]
            log_var: Log variance of latent distribution [B, latent_dim]
        Returns:
            z: Sampled latent vectors [B, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x: Input images [B, C, H, W]
        Returns:
            x_recon: Reconstructed images [B, C, H, W]
            mu: Mean of latent distribution [B, latent_dim]
            log_var: Log variance of latent distribution [B, latent_dim]
        """
        # Encode
        mu, log_var = self.encoder(x)

        # Sample z using reparameterization trick
        z = self.reparameterize(mu, log_var)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, log_var

    def sample(self, n_samples, device):
        """
        Generate new samples by sampling from the prior p(z) = N(0, I).

        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
        Returns:
            samples: Generated images [n_samples, C, H, W]
        """
        # Sample from prior
        z = torch.randn(n_samples, self.latent_dim, device=device)

        # Decode
        samples = self.decoder(z)

        return samples

    def reconstruct(self, x):
        """
        Reconstruct input images (useful for visualization).

        Args:
            x: Input images [B, C, H, W]
        Returns:
            x_recon: Reconstructed images [B, C, H, W]
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon
