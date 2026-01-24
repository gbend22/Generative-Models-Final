"""
Sampling script for NCSN
Generate images from a trained model
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from src.models.NCSN import NCSN, ExponentialMovingAverage
from src.samplers.langevin import get_sampler
from src.utils.visualization import visualize_samples, visualize_denoising_process


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']

    # Create model
    model = NCSN(config).to(device)

    # Load model weights
    if 'ema_state_dict' in checkpoint:
        print("Loading EMA weights...")
        # Load EMA weights into model
        ema = ExponentialMovingAverage(model)
        ema.shadow = checkpoint['ema_state_dict']
        ema.apply_shadow()
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}, "
          f"step {checkpoint.get('step', 'unknown')}")

    return model, config


def sample(args):
    """Generate samples from trained model"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Create sampler
    sampler = get_sampler(config, model.sigmas.cpu().numpy())

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")

    all_samples = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(num_batches), desc='Generating batches'):
        batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)

        with torch.no_grad():
            if args.show_process:
                # Generate with intermediate steps
                samples, intermediate = sampler.sample_progressive(
                    model, batch_size, device
                )

                # Save denoising process for first batch
                if i == 0:
                    process_path = os.path.join(args.output_dir, 'denoising_process.png')
                    visualize_denoising_process(
                        intermediate,
                        model.sigmas.cpu().numpy(),
                        process_path,
                        num_samples=min(8, batch_size)
                    )
                    print(f"Saved denoising process to {process_path}")
            else:
                # Standard sampling
                samples = sampler.sample(model, batch_size, device, verbose=False)

        all_samples.append(samples.cpu())

    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]

    print(f"\nGenerated {all_samples.shape[0]} samples")

    # Save samples as grid
    grid_path = os.path.join(args.output_dir, 'samples_grid.png')
    visualize_samples(all_samples, grid_path, nrow=args.nrow)
    print(f"Saved sample grid to {grid_path}")

    # Save individual samples if requested
    if args.save_individual:
        individual_dir = os.path.join(args.output_dir, 'individual')
        os.makedirs(individual_dir, exist_ok=True)

        from torchvision.utils import save_image
        from src.datasets.cifar10 import denormalize

        for i in range(all_samples.shape[0]):
            img_path = os.path.join(individual_dir, f'sample_{i:05d}.png')
            save_image(denormalize(all_samples[i]), img_path)

        print(f"Saved individual samples to {individual_dir}")

    # Save as numpy array if requested
    if args.save_numpy:
        numpy_path = os.path.join(args.output_dir, 'samples.npy')
        np.save(numpy_path, all_samples.numpy())
        print(f"Saved samples as numpy array to {numpy_path}")

    print("\nSampling complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples from trained NCSN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for sampling')
    parser.add_argument('--output-dir', type=str, default='./samples',
                        help='Output directory for samples')
    parser.add_argument('--nrow', type=int, default=8,
                        help='Number of images per row in grid')
    parser.add_argument('--show-process', action='store_true',
                        help='Visualize the denoising process')
    parser.add_argument('--save-individual', action='store_true',
                        help='Save individual sample images')
    parser.add_argument('--save-numpy', action='store_true',
                        help='Save samples as numpy array')

    args = parser.parse_args()

    sample(args)