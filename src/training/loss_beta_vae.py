import torch
import torch.nn.functional as F


def gaussian_nll_loss(x_recon, x):
    """
    Gaussian negative log-likelihood loss for reconstruction.
    Assumes fixed variance (sigma=1), which simplifies to MSE.

    Args:
        x_recon: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
    Returns:
        Reconstruction loss (summed over pixels, averaged over batch)
    """
    # MSE loss summed over all pixels
    recon_loss = F.mse_loss(x_recon, x, reduction='none')
    # Sum over C, H, W dimensions, then mean over batch
    recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1)
    return recon_loss.mean()


def bernoulli_nll_loss(x_recon, x):
    """
    Bernoulli negative log-likelihood loss for reconstruction.
    Uses binary cross-entropy.

    Args:
        x_recon: Reconstructed images [B, C, H, W] (values in [0, 1])
        x: Original images [B, C, H, W] (values in [0, 1])
    Returns:
        Reconstruction loss (summed over pixels, averaged over batch)
    """
    # Binary cross-entropy loss
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='none')
    # Sum over C, H, W dimensions, then mean over batch
    recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1)
    return recon_loss.mean()


def kl_divergence(mu, log_var):
    """
    KL divergence between the approximate posterior q(z|x) = N(mu, sigma^2)
    and the prior p(z) = N(0, I).

    KL(q||p) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu: Mean of the approximate posterior [B, latent_dim]
        log_var: Log variance of the approximate posterior [B, latent_dim]
    Returns:
        KL divergence (summed over latent dims, averaged over batch)
    """
    # KL divergence for each latent dimension
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    # Sum over latent dimensions, mean over batch
    kl = kl.sum(dim=1)
    return kl.mean()


def beta_vae_loss(x_recon, x, mu, log_var, beta=1.0, recon_type='gaussian'):
    """
    Beta-VAE loss function.

    L = E[log p(x|z)] - beta * D_KL(q(z|x) || p(z))

    Which is equivalent to minimizing:
    L = Reconstruction_Loss + beta * KL_Divergence

    Args:
        x_recon: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
        mu: Mean of latent distribution [B, latent_dim]
        log_var: Log variance of latent distribution [B, latent_dim]
        beta: Weight for KL divergence term (beta > 1 for disentanglement)
        recon_type: Type of reconstruction loss ('gaussian' or 'bernoulli')

    Returns:
        total_loss: Total beta-VAE loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component (unweighted)
    """
    # Reconstruction loss
    if recon_type == 'gaussian':
        recon_loss = gaussian_nll_loss(x_recon, x)
    elif recon_type == 'bernoulli':
        recon_loss = bernoulli_nll_loss(x_recon, x)
    else:
        raise ValueError(f"Unknown reconstruction type: {recon_type}")

    # KL divergence
    kl_loss = kl_divergence(mu, log_var)

    # Total loss with beta weighting
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
