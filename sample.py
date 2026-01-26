"""
Generate samples using trained NCSN
sample.py

Usage:
    python sample.py --checkpoint checkpoints/checkpoint_epoch_200.pth --n_samples 1000
"""

import torch
import argparse
import os
from tqdm import tqdm

from models.ncsn import NCSN
from utils.visualization import save_sample_grid, visualize_samples


def load_model(checkpoint_path, device):
    """Load trained NCSN model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Initialize model
    model = NCSN(config).to(device)

    # Load EMA weights if available (better quality)
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("✓ Loaded EMA model weights")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded model weights")

    model.eval()

    return model, config


def generate_samples(model, n_samples, device, config, batch_size=64):
    """
    Generate samples using Annealed Langevin Dynamics

    Args:
        model: Trained NCSN
        n_samples: Number of samples to generate
        device: Device to use
        config: Model config
        batch_size: Batch size for generation

    Returns:
        all_samples: Generated images [n_samples, 3, 32, 32]
    """
    all_samples = []

    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"\nGenerating {n_samples} samples...")

    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            current_batch_size = min(batch_size, n_samples - i * batch_size)

            # Generate batch
            samples = model.sample(
                batch_size=current_batch_size,
                device=device,
                n_steps_each=config.get('sample_steps', 100),
                step_lr=config.get('sample_step_lr', 2e-5)
            )

            all_samples.append(samples.cpu())

    # Concatenate all batches
    all_samples = torch.cat(all_samples, dim=0)

    return all_samples


def main(args):
    """Main sampling function"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)

    print(f"\nModel configuration:")
    print(f"  Noise levels: {config.get('num_noise_levels', 10)}")
    print(f"  Sample steps per level: {config.get('sample_steps', 100)}")
    print(f"  Base channels: {config.get('base_channels', 128)}")

    # Generate samples
    samples = generate_samples(
        model,
        n_samples=args.n_samples,
        device=device,
        config=config,
        batch_size=args.batch_size
    )

    print(f"\n✓ Generated {len(samples)} samples")
    print(f"  Shape: {samples.shape}")
    print(f"  Range: [{samples.min():.2f}, {samples.max():.2f}]")

    # Save samples
    os.makedirs(args.output_dir, exist_ok=True)

    # Save grid visualization
    grid_path = os.path.join(args.output_dir, 'samples_grid.png')
    save_sample_grid(samples[:64], grid_path, nrow=8)
    print(f"\n✓ Saved grid: {grid_path}")

    # Save all samples as tensor
    if args.save_tensor:
        tensor_path = os.path.join(args.output_dir, 'samples.pt')
        torch.save(samples, tensor_path)
        print(f"✓ Saved tensor: {tensor_path}")

    # Visualize
    if args.visualize:
        print("\nVisualizing samples...")
        visualize_samples(samples[:64], title=f"Generated Samples (n={args.n_samples})", nrow=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate samples using NCSN')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='generated_samples',
                       help='Directory to save samples')
    parser.add_argument('--save_tensor', action='store_true',
                       help='Save samples as .pt tensor file')
    parser.add_argument('--visualize', action='store_true',
                       help='Display samples using matplotlib')

    args = parser.parse_args()

    main(args)