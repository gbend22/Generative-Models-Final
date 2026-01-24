"""
Visualization utilities for NCSN
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


def denormalize(tensor):
    """Denormalize from [-1, 1] to [0, 1]"""
    return (tensor + 1.0) / 2.0


def save_image_grid(images, path, nrow=8, normalize=True):
    """
    Save a grid of images

    Args:
        images: (B, C, H, W) tensor
        path: save path
        nrow: number of images per row
        normalize: if True, denormalize from [-1,1] to [0,1]
    """
    if normalize:
        images = denormalize(images)

    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid = grid.cpu().numpy().transpose(1, 2, 0)
    grid = np.clip(grid, 0, 1)

    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_samples(samples, save_path, title='Generated Samples', nrow=8):
    """
    Visualize and save generated samples

    Args:
        samples: (B, C, H, W) tensor in [-1, 1]
        save_path: path to save figure
        title: plot title
        nrow: images per row
    """
    samples = denormalize(samples)
    grid = make_grid(samples, nrow=nrow, padding=2)
    grid = grid.cpu().numpy().transpose(1, 2, 0)

    plt.figure(figsize=(15, 15))
    plt.imshow(grid)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_denoising_process(samples_list, sigmas, save_path, num_samples=8):
    """
    Visualize the progressive denoising process

    Args:
        samples_list: list of tensors, one for each noise level
        sigmas: array of noise levels
        save_path: path to save figure
        num_samples: number of samples to show
    """
    num_levels = len(samples_list)

    fig, axes = plt.subplots(num_samples, num_levels, figsize=(num_levels * 2, num_samples * 2))

    for i in range(num_samples):
        for j, samples in enumerate(samples_list):
            img = denormalize(samples[i]).cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)

            ax = axes[i, j] if num_samples > 1 else axes[j]
            ax.imshow(img)
            ax.axis('off')

            if i == 0:
                ax.set_title(f'σ={sigmas[j]:.3f}', fontsize=10)

    plt.suptitle('Progressive Denoising', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(metrics_dict, save_path):
    """
    Plot training metrics

    Args:
        metrics_dict: dict with keys like 'loss', 'fid', etc.
        save_path: path to save figure
    """
    num_metrics = len(metrics_dict)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 4))

    if num_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics_dict.items()):
        ax.plot(values)
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_score_field(model, device, sigma_idx=0, grid_size=20, save_path=None):
    """
    Visualize the learned score field for 2D slices
    Useful for debugging

    Args:
        model: trained NCSN
        device: torch device
        sigma_idx: which noise level to visualize
        grid_size: resolution of visualization grid
        save_path: path to save figure
    """
    model.eval()

    # Create a grid of points
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x, y)

    # For simplicity, we'll just look at the first channel
    # and set other dimensions to zero
    scores_x = np.zeros_like(X)
    scores_y = np.zeros_like(Y)

    with torch.no_grad():
        for i in range(grid_size):
            for j in range(grid_size):
                # Create input (simplified 2D case)
                point = torch.zeros(1, 3, 32, 32, device=device)
                point[0, 0, 16, 16] = X[i, j]
                point[0, 1, 16, 16] = Y[i, j]

                labels = torch.tensor([sigma_idx], device=device)
                score = model(point, labels)

                scores_x[i, j] = score[0, 0, 16, 16].cpu().item()
                scores_y[i, j] = score[0, 1, 16, 16].cpu().item()

    # Plot vector field
    plt.figure(figsize=(10, 10))
    plt.quiver(X, Y, scores_x, scores_y, alpha=0.6)
    plt.title(f'Score Field (σ_idx={sigma_idx})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_real_fake(real_images, fake_images, save_path, num_samples=16):
    """
    Create side-by-side comparison of real and generated images

    Args:
        real_images: (B, C, H, W) real images
        fake_images: (B, C, H, W) generated images
        save_path: path to save figure
        num_samples: number of samples to show
    """
    real_images = denormalize(real_images[:num_samples])
    fake_images = denormalize(fake_images[:num_samples])

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    for i in range(num_samples):
        # Real images
        real_img = real_images[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(np.clip(real_img, 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Real', fontsize=12)

        # Fake images
        fake_img = fake_images[i].cpu().numpy().transpose(1, 2, 0)
        axes[1, i].imshow(np.clip(fake_img, 0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Generated', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Test visualization functions
    print("Testing visualization utilities...")

    # Create dummy samples
    samples = torch.randn(64, 3, 32, 32)

    os.makedirs('test_vis', exist_ok=True)

    # Test image grid
    save_image_grid(samples, 'test_vis/grid.png', nrow=8)
    print("Saved: test_vis/grid.png")

    # Test sample visualization
    visualize_samples(samples, 'test_vis/samples.png')
    print("Saved: test_vis/samples.png")

    # Test training curves
    metrics = {
        'loss': np.random.randn(100).cumsum(),
        'fid': np.abs(np.random.randn(100).cumsum()),
    }
    plot_training_curves(metrics, 'test_vis/curves.png')
    print("Saved: test_vis/curves.png")

    print("\nVisualization tests complete!")