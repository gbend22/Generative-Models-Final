"""
Sampling script for NCSN.

Usage:
    python scripts/sample.py --checkpoint path/to/checkpoint.pt --num_samples 64

    # Save progressive samples
    python scripts/sample.py --checkpoint path/to/checkpoint.pt --progressive

    # Use specific sampling parameters
    python scripts/sample.py --checkpoint path/to/checkpoint.pt \\
        --num_steps 200 --step_lr 1e-5 --denoise

This script loads a trained NCSN model and generates samples using
Annealed Langevin Dynamics.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ncsn.model import NCSN, get_model
from ncsn.sampler import AnnealedLangevinDynamics, sample_from_model
from ncsn.utils import (
    get_sigmas,
    EMA,
    load_checkpoint,
    make_grid,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Sample from trained NCSN')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./generated',
                        help='Output directory for samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    # Sampling parameters
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Number of Langevin steps per noise level')
    parser.add_argument('--step_lr', type=float, default=None,
                        help='Step size for Langevin dynamics')
    parser.add_argument('--denoise', action='store_true',
                        help='Apply denoising at final step')
    parser.add_argument('--no_denoise', action='store_true',
                        help='Do not apply denoising')

    # Options
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA weights for sampling')
    parser.add_argument('--progressive', action='store_true',
                        help='Save progressive samples')
    parser.add_argument('--nrow', type=int, default=8,
                        help='Number of images per row in grid')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    # Compute noise levels
    sigmas = get_sigmas(
        sigma_begin=config['noise']['sigma_begin'],
        sigma_end=config['noise']['sigma_end'],
        num_classes=config['model']['num_classes'],
    ).to(device)

    # Create model
    model = get_model(config, sigmas).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load EMA if available
    if args.use_ema and 'ema_state_dict' in checkpoint:
        print("Using EMA weights")
        ema = EMA(model, decay=checkpoint['ema_state_dict']['decay'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.apply_shadow()

    # Determine sampling parameters
    num_steps = args.num_steps or config['sampling']['num_steps']
    step_lr = args.step_lr or config['sampling']['step_lr']
    denoise = config['sampling']['denoise']
    if args.denoise:
        denoise = True
    if args.no_denoise:
        denoise = False

    print(f"\nSampling parameters:")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Num steps per noise level: {num_steps}")
    print(f"  Step LR: {step_lr}")
    print(f"  Denoise: {denoise}")
    print(f"  Noise levels: {len(sigmas)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create sampler
    sampler = AnnealedLangevinDynamics(
        score_net=model,
        sigmas=sigmas,
        num_steps=num_steps,
        step_lr=step_lr,
        denoise=denoise,
        device=device,
    )

    if args.progressive:
        # Generate progressive samples
        print("\nGenerating progressive samples...")
        intermediates = sampler.sample_progressive(
            batch_size=min(args.num_samples, 16),  # Fewer samples for progressive
            img_shape=(3, 32, 32),
            save_freq=max(1, len(sigmas) // 10),  # Save ~10 intermediate steps
        )

        # Save each intermediate step
        for i, samples in enumerate(intermediates):
            grid = make_grid(samples, nrow=4, normalize=True)
            save_path = os.path.join(args.output_dir, f'progressive_{i:03d}.png')
            torchvision.utils.save_image(grid, save_path)

        print(f"Saved {len(intermediates)} progressive steps to {args.output_dir}")

        # Create GIF
        try:
            from PIL import Image
            import glob

            frames = []
            for path in sorted(glob.glob(os.path.join(args.output_dir, 'progressive_*.png'))):
                frames.append(Image.open(path))

            if frames:
                gif_path = os.path.join(args.output_dir, 'sampling_process.gif')
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=200,
                    loop=0
                )
                print(f"Saved GIF to {gif_path}")
        except ImportError:
            print("PIL not available, skipping GIF creation")

    else:
        # Generate final samples
        print("\nGenerating samples...")
        samples = sampler.sample(
            batch_size=args.num_samples,
            img_shape=(3, 32, 32),
            verbose=True,
        )

        # Save grid
        grid = make_grid(samples, nrow=args.nrow, normalize=True)
        grid_path = os.path.join(args.output_dir, 'samples_grid.png')
        torchvision.utils.save_image(grid, grid_path)
        print(f"Saved sample grid to {grid_path}")

        # Save individual samples
        individual_dir = os.path.join(args.output_dir, 'individual')
        os.makedirs(individual_dir, exist_ok=True)

        # Unnormalize samples from [-1, 1] to [0, 1]
        samples_unnorm = (samples.clamp(-1, 1) + 1) / 2

        for i in range(samples.shape[0]):
            img_path = os.path.join(individual_dir, f'sample_{i:04d}.png')
            torchvision.utils.save_image(samples_unnorm[i], img_path)

        print(f"Saved {args.num_samples} individual samples to {individual_dir}")

    print("\nDone!")


if __name__ == '__main__':
    main()