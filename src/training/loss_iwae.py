"""
IWAE Loss Function
Burda, Grosse & Salakhutdinov (2016)

The IWAE objective uses k importance-weighted samples for a tighter
log-likelihood bound than the standard VAE ELBO:

L_k(x) = E[log(1/k * sum_{i=1}^{k} w_i)]

where w_i = p(x, z_i) / q(z_i|x)
"""
import torch
import torch.nn.functional as F
import math


LOG_2PI = math.log(2 * math.pi)


def log_normal_pdf(z, mu, log_var):
    """
    Log probability of z under diagonal Gaussian N(mu, diag(exp(log_var))).

    Args:
        z: [B, k, D] or [B, D]
        mu: [B, D]
        log_var: [B, D]
    Returns:
        log_prob: [B, k] or [B] -- summed over D dimensions
    """
    # If z is [B, k, D], expand mu and log_var to [B, 1, D]
    if z.dim() == 3:
        mu = mu.unsqueeze(1)  # [B, 1, D]
        log_var = log_var.unsqueeze(1)  # [B, 1, D]

    D = z.shape[-1]
    # log N(z; mu, sigma^2) = -0.5 * [D*log(2pi) + sum(log_var) + sum((z-mu)^2/var)]
    log_prob = -0.5 * (
        D * LOG_2PI
        + log_var.sum(dim=-1)
        + ((z - mu).pow(2) / log_var.exp()).sum(dim=-1)
    )
    return log_prob  # [B, k] or [B]


def log_standard_normal_pdf(z):
    """
    Log probability of z under standard normal N(0, I).

    Args:
        z: [B, k, D]
    Returns:
        log_prob: [B, k] -- summed over D dimensions
    """
    D = z.shape[-1]
    # log N(z; 0, I) = -0.5 * [D*log(2pi) + sum(z^2)]
    log_prob = -0.5 * (D * LOG_2PI + z.pow(2).sum(dim=-1))
    return log_prob  # [B, k]


def iwae_loss(x_recon, x, mu, log_var, z, k, recon_type='gaussian'):
    """
    IWAE loss: -L_k(x) = -E[log(1/k * sum w_i)]

    Args:
        x_recon: Reconstructed images [B*k, C, H, W]
        x: Original images [B, C, H, W]
        mu: Encoder mean [B, latent_dim]
        log_var: Encoder log variance [B, latent_dim]
        z: Sampled latents [B, k, latent_dim]
        k: Number of importance samples
        recon_type: 'gaussian' or 'bernoulli'
    Returns:
        loss: Scalar IWAE loss (averaged over batch)
        recon_loss_avg: Average reconstruction loss (for logging)
        kl_loss_avg: Average KL divergence approximation (for logging)
    """
    B = x.shape[0]
    C, H, W = x.shape[1], x.shape[2], x.shape[3]

    # --- Step 1: Compute log p(x|z_i) for each sample ---
    # Expand x to match x_recon: [B, C, H, W] -> [B*k, C, H, W]
    x_expanded = x.unsqueeze(1).expand(B, k, C, H, W).reshape(B * k, C, H, W)

    if recon_type == 'gaussian':
        # Gaussian NLL (assuming unit variance): -log p(x|z) = 0.5 * ||x - x_recon||^2
        # We compute per-sample (not averaged over batch)
        recon_diff = (x_recon - x_expanded).pow(2)
        nll = recon_diff.view(B * k, -1).sum(dim=1)  # [B*k]
    elif recon_type == 'bernoulli':
        # Binary cross-entropy
        bce = F.binary_cross_entropy(x_recon, x_expanded, reduction='none')
        nll = bce.view(B * k, -1).sum(dim=1)  # [B*k]
    else:
        raise ValueError(f"Unknown recon_type: {recon_type}")

    log_p_x_given_z = -0.5 * nll.reshape(B, k)  # [B, k], scaled by 0.5 for Gaussian

    # --- Step 2: Compute log p(z_i) (standard normal prior) ---
    log_p_z = log_standard_normal_pdf(z)  # [B, k]

    # --- Step 3: Compute log q(z_i|x) (encoder distribution) ---
    log_q_z_given_x = log_normal_pdf(z, mu, log_var)  # [B, k]

    # --- Step 4: Compute log importance weights ---
    log_w = log_p_x_given_z + log_p_z - log_q_z_given_x  # [B, k]

    # --- Step 5: IWAE objective using logsumexp for numerical stability ---
    # L_k = E[logsumexp(log_w) - log(k)]
    iwae_bound = torch.logsumexp(log_w, dim=1) - math.log(k)  # [B]

    # Loss is negative IWAE bound (we minimize)
    loss = -iwae_bound.mean()

    # --- Logging quantities (for monitoring, comparable to VAE metrics) ---
    # Average reconstruction NLL across all samples
    recon_loss_avg = nll.mean()
    # Approximate KL as mean(log_q - log_p) across samples
    kl_approx = (log_q_z_given_x - log_p_z).mean()

    return loss, recon_loss_avg, kl_approx
