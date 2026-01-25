import torch

def denoising_score_matching_loss(model, x, sigmas):
    batch_size = x.size(0)
    device = x.device

    sigma_idx = torch.randint(
        0, len(sigmas), (batch_size,), device=device
    )

    sigma = sigmas[sigma_idx].view(batch_size, 1, 1, 1)
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise

    score = model(x_noisy, sigma_idx)

    loss = torch.mean(
        torch.sum((score + noise / sigma) ** 2, dim=(1, 2, 3))
    )

    return loss
