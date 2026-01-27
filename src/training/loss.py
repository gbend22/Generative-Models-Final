import torch


def denoising_score_matching_loss(score_net, x, sigmas):
    """
    Computes the Denoising Score Matching loss.

    Args:
        score_net: The U-Net model.
        x: Batch of clean images [Batch, 3, 32, 32].
        sigmas: The list of all sigma values (geometric sequence).
    """
    batch_size = x.shape[0]

    # 1. Randomly sample a sigma index for each image in the batch
    # We want to train on ALL noise levels simultaneously.
    sigma_indices = torch.randint(0, len(sigmas), (batch_size,), device=x.device)
    current_sigmas = sigmas[sigma_indices]  # Shape: [Batch]

    # Reshape sigmas for broadcasting [Batch, 1, 1, 1]
    sigmas_reshaped = current_sigmas.view(batch_size, 1, 1, 1)

    # 2. Perturb the data (Add Noise)
    # x_tilde = x + sigma * z
    z = torch.randn_like(x)  # Standard Gaussian noise
    x_tilde = x + sigmas_reshaped * z

    # 3. Predict the Score
    # The network takes the noisy image and the specific sigma used
    predicted_score = score_net(x_tilde, current_sigmas)

    # 4. Calculate Target
    # The score of N(x, sigma^2) is -(x_tilde - x) / sigma^2 = -z / sigma
    target_score = -z / sigmas_reshaped

    # 5. Calculate Loss
    # We weight the loss by sigma^2 to keep the magnitude balanced across scales
    # Loss = (1/2) * lambda(sigma) * || score - target ||^2
    # lambda(sigma) = sigma^2

    diff = predicted_score - target_score

    # Sum over pixel dimensions (C, H, W) but average over Batch
    squared_diff = diff ** 2
    # || s - target ||^2
    l2_norm = torch.sum(squared_diff.view(batch_size, -1), dim=1)

    # Apply the weighting: sigma^2 * l2_norm
    # Note: 0.5 factor is standard in the derivation but often omitted in code
    # as it just scales the learning rate. We'll include it for rigor.
    loss = 0.5 * (current_sigmas ** 2) * l2_norm

    return loss.mean()