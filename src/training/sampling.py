import torch
import numpy as np
from tqdm import tqdm


def annealed_langevin_dynamics(model, sigmas, config, n_samples=16):
    """
    Implements Algorithm 1 from Song & Ermon (2019).

    Args:
        model: The trained ScoreNet.
        sigmas: Tensor of noise levels (sigma_1 > ... > sigma_L).
        config: Configuration object containing step_lr (epsilon) and n_steps_per_scale (T).
        n_samples: Number of images to generate.
    """
    device = config.device
    model.eval()

    # 1. Initialize with uniform random noise in [0, 1] range
    # Paper uses uniform noise as initial samples
    x = torch.rand(n_samples, config.channels, config.image_size, config.image_size, device=device)

    # 2. Iterate through noise levels (Annealing)
    with torch.no_grad():
        for i, sigma in enumerate(tqdm(sigmas, desc="Sampling")):

            # Helper to broadcast sigma to batch size
            sigma_batch = torch.ones(n_samples, device=device) * sigma

            # Calculate step size for this noise level
            # alpha_i = epsilon * (sigma_i / sigma_L)^2
            # This maintains a constant Signal-to-Noise Ratio (SNR)
            step_size = config.step_lr * (sigma / sigmas[-1]) ** 2

            # 3. Langevin Dynamics (Inner Loop)
            for t in range(config.n_steps_per_scale):
                # Sample noise z
                z = torch.randn_like(x)

                # Predict Score s(x, sigma)
                score = model(x, sigma_batch)

                # Clip score to prevent explosion (numerical stability)
                score = torch.clamp(score, -1e4, 1e4)

                # Update x using Langevin Dynamics Rule:
                # x_{t+1} = x_t + (step_size/2) * score + sqrt(step_size) * z
                gradient_term = 0.5 * step_size * score
                noise_term = np.sqrt(step_size) * z

                x = x + gradient_term + noise_term

    # 4. Final step: clamp to valid image range [0, 1]
    return torch.clamp(x, 0.0, 1.0)