import torch
import numpy as np
from tqdm import tqdm


def annealed_langevin_dynamics(model, sigmas, config, n_samples=16):
    device = config.device
    model.eval()

    # --- FIX 1: Initialization ---
    # Start with random noise that matches the magnitude of sigma_start
    # or simple uniform noise in [0,1]
    x = torch.rand(n_samples, config.channels, config.image_size, config.image_size, device=device)

    with torch.no_grad():
        for i, sigma in enumerate(tqdm(sigmas, desc="Sampling")):
            sigma_batch = torch.ones(n_samples, device=device) * sigma

            # Step size calculation
            step_size = config.step_lr * (sigma / sigmas[-1]) ** 2

            for t in range(config.n_steps_per_scale):
                z = torch.randn_like(x)

                # Predict Score
                score = model(x, sigma_batch)

                # Langevin Update
                noise_term = torch.sqrt(step_size) * z
                gradient_term = 0.5 * step_size * score

                x = x + gradient_term + noise_term

    # --- FIX 2: Better Clamping ---
    # Ensure we don't return crazy values
    return torch.clamp(x, 0.0, 1.0)