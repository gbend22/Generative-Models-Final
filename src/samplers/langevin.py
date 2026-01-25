import torch

@torch.no_grad()
def annealed_langevin_sampling(
    model,
    sigmas,
    n_steps=100,
    step_lr=0.00002,
    shape=(64, 3, 32, 32),
    device="cuda"
):
    x = torch.randn(shape, device=device)

    for sigma_idx, sigma in enumerate(sigmas):
        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for _ in range(n_steps):
            grad = model(x, torch.full(
                (shape[0],), sigma_idx, device=device, dtype=torch.long
            ))
            noise = torch.randn_like(x)
            x = x + step_size * grad + torch.sqrt(2 * step_size) * noise

    return x
