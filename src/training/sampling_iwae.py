"""
IWAE Sampling Utilities
"""
import torch
import torch.nn.functional as F
import numpy as np


def sample_from_prior(model, n_samples, device):
    """
    Generate samples from prior p(z) = N(0, I).

    Args:
        model: Trained IWAE model
        n_samples: Number of samples to generate
        device: Device to use
    Returns:
        Generated images [n_samples, C, H, W]
    """
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples, device)
    return samples


def reconstruct_images(model, images, device):
    """
    Reconstruct images using a single sample.

    Args:
        model: Trained IWAE model
        images: Input images [B, C, H, W]
        device: Device to use
    Returns:
        Reconstructions [B, C, H, W]
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        reconstructions = model.reconstruct(images)
    return reconstructions


def iwae_reconstruct_best(model, images, device, k=50):
    """
    Reconstruct using k samples and select the best one
    (highest importance weight) per image.

    This is IWAE-specific: we can pick the reconstruction
    corresponding to the highest log w_i.

    Args:
        model: Trained IWAE model
        images: [B, C, H, W]
        device: Device
        k: Number of samples to draw
    Returns:
        best_recons: [B, C, H, W]
    """
    from src.training.loss_iwae import log_standard_normal_pdf, log_normal_pdf

    model.eval()
    with torch.no_grad():
        images = images.to(device)
        x_recon, mu, log_var, z = model(images, k=k)

        B = images.shape[0]
        C, H, W = images.shape[1], images.shape[2], images.shape[3]

        # Compute log importance weights to find best sample
        # log p(x|z_i)
        x_expanded = images.unsqueeze(1).expand(B, k, C, H, W).reshape(B * k, C, H, W)
        recon_diff = (x_recon - x_expanded).pow(2)
        nll = recon_diff.view(B * k, -1).sum(dim=1).reshape(B, k)
        log_p_x_given_z = -0.5 * nll

        # log p(z_i), log q(z_i|x)
        log_p_z = log_standard_normal_pdf(z)
        log_q_z_given_x = log_normal_pdf(z, mu, log_var)

        log_w = log_p_x_given_z + log_p_z - log_q_z_given_x  # [B, k]
        best_idx = log_w.argmax(dim=1)  # [B]

        # Gather best reconstructions
        x_recon_reshaped = x_recon.reshape(B, k, C, H, W)
        best_recons = x_recon_reshaped[torch.arange(B, device=device), best_idx]

    return best_recons


def latent_traversal(model, image, device, latent_idx, range_vals=(-3, 3), n_steps=10):
    """
    Traverse a single latent dimension while keeping others fixed.

    Args:
        model: Trained IWAE model
        image: Single input image [C, H, W] or [1, C, H, W]
        device: Device to use
        latent_idx: Index of latent dimension to traverse
        range_vals: Tuple (min, max) for traversal range
        n_steps: Number of steps in traversal
    Returns:
        Traversal images [n_steps, C, H, W]
    """
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        mu, _ = model.encoder(image)
        traversal_values = np.linspace(range_vals[0], range_vals[1], n_steps)
        traversal_images = []
        for val in traversal_values:
            z = mu.clone()
            z[0, latent_idx] = val
            recon = model.decoder(z)
            traversal_images.append(recon)
        traversal_images = torch.cat(traversal_images, dim=0)
    return traversal_images


def interpolate_latents(model, image1, image2, device, n_steps=10):
    """
    Interpolate between two images in latent space.

    Args:
        model: Trained IWAE model
        image1: First image [C, H, W] or [1, C, H, W]
        image2: Second image [C, H, W] or [1, C, H, W]
        device: Device to use
        n_steps: Number of interpolation steps
    Returns:
        Interpolated images [n_steps, C, H, W]
    """
    model.eval()
    if image1.dim() == 3:
        image1 = image1.unsqueeze(0)
    if image2.dim() == 3:
        image2 = image2.unsqueeze(0)
    image1 = image1.to(device)
    image2 = image2.to(device)

    with torch.no_grad():
        mu1, _ = model.encoder(image1)
        mu2, _ = model.encoder(image2)
        interpolations = []
        for alpha in np.linspace(0, 1, n_steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            recon = model.decoder(z)
            interpolations.append(recon)
        interpolations = torch.cat(interpolations, dim=0)
    return interpolations
