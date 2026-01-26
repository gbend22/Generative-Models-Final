"""
Visualization utilities for NCSN
utils/visualization.py
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os


def denormalize(images):
    """
    Denormalize images from [-1, 1] to [0, 1]

    Args:
        images: Tensor of shape [B, 3, H, W] in range [-1, 1]

    Returns:
        Denormalized images in range [0, 1]
    """
    return (images + 1) / 2


def save_sample_grid(samples, save_path, nrow=8, padding=2):
    """
    Save a grid of generated samples

    Args:
        samples: Generated images [B, 3, H, W] in range [-1, 1]
        save_path: Where to save the image
        nrow: Number of images per row
        padding: Padding between images
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Denormalize
    samples = denormalize(samples)

    # Clamp to valid range
    samples = torch.clamp(samples, 0, 1)

    # Create grid
    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=padding)

    # Save
    torchvision.utils.save_image(grid, save_path)


def visualize_samples(samples, title="Generated Samples", nrow=8):
    """
    Display samples using matplotlib

    Args:
        samples: Generated images [B, 3, H, W]
        title: Plot title
        nrow: Number of images per row
    """
    samples = denormalize(samples.cpu())
    samples = torch.clamp(samples, 0, 1)

    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(15, 15))
    plt.imshow(grid_np)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_noise_levels(model, x_clean, save_path=None):
    """
    Visualize how images look at different noise levels

    Args:
        model: NCSN model
        x_clean: Clean image [1, 3, 32, 32]
        save_path: Optional path to save visualization
    """
    n_levels = len(model.sigmas)

    fig, axes = plt.subplots(2, (n_levels + 1) // 2, figsize=(15, 6))
    axes = axes.flatten()

    x_clean_np = denormalize(x_clean).squeeze(0).permute(1, 2, 0).cpu().numpy()

    for i, sigma in enumerate(model.sigmas):
        # Add noise
        noise = torch.randn_like(x_clean)
        x_noisy = x_clean + sigma * noise

        # Denormalize and convert to numpy
        x_noisy_np = denormalize(x_noisy).squeeze(0).permute(1, 2, 0).cpu().numpy()
        x_noisy_np = np.clip(x_noisy_np, 0, 1)

        # Plot
        axes[i].imshow(x_noisy_np)
        axes[i].set_title(f'σ = {sigma:.4f}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle('Image at Different Noise Levels', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_sampling_process(model, device, save_path=None):
    """
    Visualize the annealed Langevin dynamics sampling process

    Shows intermediate samples as we anneal from high to low noise
    """
    model.eval()

    # Start from noise
    x = torch.randn(1, 3, 32, 32, device=device)

    # Store intermediate samples
    intermediates = [x.cpu()]
    sigma_indices = []

    n_steps_each = 50  # Fewer steps for visualization

    with torch.no_grad():
        for sigma_idx in range(len(model.sigmas)):
            sigma = model.sigmas[sigma_idx]
            labels = torch.ones(1, device=device, dtype=torch.long) * sigma_idx

            step_size = 2e-5 * (sigma / model.sigmas[-1]) ** 2

            # Run a few Langevin steps
            for step in range(n_steps_each):
                noise = torch.randn_like(x)
                scores, _ = model(x, labels)
                x = x + step_size * scores + torch.sqrt(2 * step_size) * noise
                x = torch.clamp(x, -1, 1)

                # Store every 10 steps
                if step % 10 == 0:
                    intermediates.append(x.cpu())
                    sigma_indices.append(sigma_idx)

    # Plot
    n_show = min(20, len(intermediates))
    indices = np.linspace(0, len(intermediates) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show // 2, figsize=(20, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img = denormalize(intermediates[idx]).squeeze(0).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        if idx < len(sigma_indices):
            axes[i].set_title(f'σ level {sigma_indices[idx]}', fontsize=8)
        else:
            axes[i].set_title('Final', fontsize=8)
        axes[i].axis('off')

    plt.suptitle('Sampling Process: High Noise → Low Noise', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def compare_real_vs_generated(real_images, generated_images, save_path=None):
    """
    Side-by-side comparison of real vs generated images

    Args:
        real_images: Real CIFAR-10 images [B, 3, 32, 32]
        generated_images: Generated images [B, 3, 32, 32]
        save_path: Optional path to save
    """
    n = min(8, real_images.shape[0], generated_images.shape[0])

    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))

    real = denormalize(real_images[:n]).cpu()
    generated = denormalize(generated_images[:n]).cpu()

    for i in range(n):
        # Real
        axes[0, i].imshow(real[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title('Real', fontsize=10)
        axes[0, i].axis('off')

        # Generated
        axes[1, i].imshow(torch.clamp(generated[i], 0, 1).permute(1, 2, 0).numpy())
        axes[1, i].set_title('Generated', fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_training_curves(losses, save_path=None):
    """
    Plot training loss curves

    Args:
        losses: List of loss values
        save_path: Optional path to save
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


# Test visualization functions
if __name__ == "__main__":
    print("Testing visualization utilities...")

    # Generate random samples
    samples = torch.randn(64, 3, 32, 32)

    # Test save_sample_grid
    print("Testing save_sample_grid...")
    save_sample_grid(samples, 'test_samples.png', nrow=8)
    print("✓ Sample grid saved to test_samples.png")

    # Test visualize_samples
    print("\nTesting visualize_samples...")
    visualize_samples(samples[:16], title="Test Samples", nrow=4)

    print("\n✓ Visualization utilities working!")