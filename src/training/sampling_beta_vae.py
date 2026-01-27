import torch
import numpy as np


def sample_from_prior(model, n_samples, device):
    """
    Generate samples by sampling from the prior p(z) = N(0, I).

    Args:
        model: Trained BetaVAE model
        n_samples: Number of samples to generate
        device: Device to use
    Returns:
        samples: Generated images [n_samples, C, H, W]
    """
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples, device)
    return samples


def reconstruct_images(model, images, device):
    """
    Reconstruct input images through the VAE.

    Args:
        model: Trained BetaVAE model
        images: Input images [B, C, H, W]
        device: Device to use
    Returns:
        reconstructions: Reconstructed images [B, C, H, W]
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        reconstructions = model.reconstruct(images)
    return reconstructions


def latent_traversal(model, image, device, latent_idx, range_vals=(-3, 3), n_steps=10):
    """
    Traverse a single latent dimension while keeping others fixed.
    This is useful for visualizing what each latent dimension encodes.

    Args:
        model: Trained BetaVAE model
        image: Input image to encode [1, C, H, W] or [C, H, W]
        device: Device to use
        latent_idx: Index of the latent dimension to traverse
        range_vals: (min, max) range for traversal
        n_steps: Number of steps in the traversal
    Returns:
        traversal_images: Images from the traversal [n_steps, C, H, W]
    """
    model.eval()

    # Ensure image has batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        # Encode the image to get mu (we use mu instead of sampling)
        mu, _ = model.encoder(image)

        # Create traversal values
        traversal_values = np.linspace(range_vals[0], range_vals[1], n_steps)

        traversal_images = []
        for val in traversal_values:
            # Copy mu and modify the specified latent dimension
            z = mu.clone()
            z[0, latent_idx] = val

            # Decode
            recon = model.decoder(z)
            traversal_images.append(recon)

        # Stack into a single tensor
        traversal_images = torch.cat(traversal_images, dim=0)

    return traversal_images


def all_latent_traversals(model, image, device, range_vals=(-3, 3), n_steps=10):
    """
    Traverse all latent dimensions one by one.

    Args:
        model: Trained BetaVAE model
        image: Input image [1, C, H, W] or [C, H, W]
        device: Device to use
        range_vals: (min, max) range for traversal
        n_steps: Number of steps in the traversal
    Returns:
        all_traversals: Dict mapping latent_idx -> traversal images
    """
    model.eval()

    latent_dim = model.latent_dim
    all_traversals = {}

    for i in range(latent_dim):
        traversal = latent_traversal(model, image, device, i, range_vals, n_steps)
        all_traversals[i] = traversal

    return all_traversals


def interpolate_latents(model, image1, image2, device, n_steps=10):
    """
    Interpolate between two images in latent space.

    Args:
        model: Trained BetaVAE model
        image1: First image [1, C, H, W] or [C, H, W]
        image2: Second image [1, C, H, W] or [C, H, W]
        device: Device to use
        n_steps: Number of interpolation steps
    Returns:
        interpolations: Interpolated images [n_steps, C, H, W]
    """
    model.eval()

    # Ensure images have batch dimension
    if image1.dim() == 3:
        image1 = image1.unsqueeze(0)
    if image2.dim() == 3:
        image2 = image2.unsqueeze(0)

    image1 = image1.to(device)
    image2 = image2.to(device)

    with torch.no_grad():
        # Encode both images
        mu1, _ = model.encoder(image1)
        mu2, _ = model.encoder(image2)

        # Interpolate in latent space
        interpolations = []
        for alpha in np.linspace(0, 1, n_steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            recon = model.decoder(z)
            interpolations.append(recon)

        interpolations = torch.cat(interpolations, dim=0)

    return interpolations
