"""
Evaluation script for NCSN
Compute FID and Inception Score
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from src.models.NCSN import NCSN, ExponentialMovingAverage
from src.samplers.langevin import get_sampler
from src.datasets.cifar10 import get_cifar10_dataloaders
from src.utils.metrics import calculate_fid, calculate_inception_score, InceptionV3


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']

    # Create model
    model = NCSN(config).to(device)

    # Load model weights (prefer EMA)
    if 'ema_state_dict' in checkpoint:
        print("Using EMA weights...")
        ema = ExponentialMovingAverage(model)
        ema.shadow = checkpoint['ema_state_dict']
        ema.apply_shadow()
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return model, config


def generate_samples(model, config, num_samples, batch_size, device):
    """Generate samples from model"""
    sampler = get_sampler(config, model.sigmas.cpu().numpy())

    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"Generating {num_samples} samples...")
    for i in tqdm(range(num_batches), desc='Generating'):
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        with torch.no_grad():
            samples = sampler.sample(model, current_batch_size, device, verbose=False)

        all_samples.append(samples.cpu())

    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    return all_samples


def get_real_images(config, num_samples):
    """Get real images from CIFAR-10 test set"""
    _, test_loader = get_cifar10_dataloaders(config, augment=False)

    all_images = []
    total = 0

    print(f"Loading {num_samples} real images...")
    for images, _ in tqdm(test_loader, desc='Loading real images'):
        all_images.append(images)
        total += images.shape[0]
        if total >= num_samples:
            break

    all_images = torch.cat(all_images, dim=0)[:num_samples]
    return all_images


def evaluate(args):
    """Evaluate trained model"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Generate samples
    fake_images = generate_samples(
        model, config,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device
    )

    # Get real images
    real_images = get_real_images(config, num_samples=args.num_samples)

    print(f"\nReal images shape: {real_images.shape}")
    print(f"Fake images shape: {fake_images.shape}")

    # Calculate FID
    if not args.skip_fid:
        print("\n" + "=" * 50)
        print("Calculating FID...")
        print("=" * 50)
        fid = calculate_fid(
            real_images, fake_images,
            batch_size=args.batch_size,
            device=device
        )
        print(f"\nFID Score: {fid:.2f}")

    # Calculate Inception Score
    if not args.skip_is:
        print("\n" + "=" * 50)
        print("Calculating Inception Score...")
        print("=" * 50)
        inception_model = InceptionV3().to(device)
        is_mean, is_std = calculate_inception_score(
            fake_images, inception_model,
            batch_size=args.batch_size,
            splits=10,
            device=device
        )
        print(f"\nInception Score: {is_mean:.2f} ± {is_std:.2f}")

    # Save results
    results = {
        'num_samples': args.num_samples,
    }

    if not args.skip_fid:
        results['fid'] = float(fid)

    if not args.skip_is:
        results['is_mean'] = float(is_mean)
        results['is_std'] = float(is_std)

    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate NCSN on CIFAR-10')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of samples for evaluation')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for evaluation')
    parser.add_argument('--skip-fid', action='store_true',
                        help='Skip FID calculation')
    parser.add_argument('--skip-is', action='store_true',
                        help='Skip Inception Score calculation')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')

    args = parser.parse_args()

    evaluate(args)