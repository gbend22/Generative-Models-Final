"""
Training script for NCSN on CIFAR-10.

Usage:
    python scripts/train.py --config configs/cifar10.yaml

    # Resume training
    python scripts/train.py --config configs/cifar10.yaml --resume path/to/checkpoint.pt

    # Override config values
    python scripts/train.py --config configs/cifar10.yaml --batch_size 64 --lr 1e-4

This script:
1. Loads configuration and creates data loaders
2. Initializes model, optimizer, and EMA
3. Trains using Denoising Score Matching loss
4. Logs metrics to Weights & Biases
5. Saves checkpoints periodically
6. Generates samples for visualization
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncsn.model import NCSN, get_model
from ncsn.loss import dsm_loss, DSMLoss
from ncsn.sampler import AnnealedLangevinDynamics, sample_from_model
from ncsn.data import get_cifar10_dataloader
from ncsn.utils import (
    get_sigmas,
    EMA,
    save_checkpoint,
    load_checkpoint,
    load_config,
    count_parameters,
    set_seed,
    make_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train NCSN on CIFAR-10')

    parser.add_argument('--config', type=str, default='configs/cifar10.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')

    # Override config values
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    # W&B settings
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')

    return parser.parse_args()


def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        sigmas: torch.Tensor,
        ema: EMA,
        device: str,
        epoch: int,
        config: dict,
        grad_clip: float = 1.0,
) -> dict:
    """
    Train for one epoch.

    Args:
        model: NCSN model
        dataloader: Training data
        optimizer: Optimizer
        sigmas: Noise levels
        ema: EMA tracker
        device: Device
        epoch: Current epoch
        config: Config dict
        grad_clip: Gradient clipping value

    Returns:
        Dict with average loss and other metrics
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (x, _) in enumerate(pbar):
        x = x.to(device)

        optimizer.zero_grad()

        # Compute DSM loss
        loss, info = dsm_loss(model, x, sigmas)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update()

        # Accumulate metrics
        total_loss += info['loss']
        total_mse += info['mse_unweighted']
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{info['loss']:.4f}",
            'mse': f"{info['mse_unweighted']:.4f}",
        })

        # Log to W&B (every N batches)
        if batch_idx % config['training']['log_interval'] == 0:
            if wandb.run is not None:
                wandb.log({
                    'train/loss': info['loss'],
                    'train/mse_unweighted': info['mse_unweighted'],
                    'train/score_norm': info['score_norm'],
                    'train/sigma_mean': info['sigma_mean'],
                    'train/step': epoch * len(dataloader) + batch_idx,
                })

    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
    }


@torch.no_grad()
def generate_samples(
        model: nn.Module,
        sigmas: torch.Tensor,
        config: dict,
        device: str,
        use_ema: bool = True,
        ema: EMA = None,
) -> torch.Tensor:
    """Generate samples for visualization."""
    model.eval()

    # Optionally use EMA weights
    if use_ema and ema is not None:
        ema.apply_shadow()

    samples = sample_from_model(
        score_net=model,
        sigmas=sigmas,
        num_samples=config['sampling']['num_samples'],
        num_steps=config['sampling']['num_steps'],
        step_lr=config['sampling']['step_lr'],
        denoise=config['sampling']['denoise'],
        device=device,
        verbose=True,
    )

    if use_ema and ema is not None:
        ema.restore()

    return samples


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Override config with command line args
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs

    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize W&B
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project or config['wandb']['project'],
            name=args.wandb_name or config['wandb']['name'],
            config=config,
        )

    # Create data loader
    train_loader = get_cifar10_dataloader(
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        augment=True,
    )
    print(f"Dataset size: {len(train_loader.dataset)}")

    # Compute noise levels
    sigmas = get_sigmas(
        sigma_begin=config['noise']['sigma_begin'],
        sigma_end=config['noise']['sigma_end'],
        num_classes=config['model']['num_classes'],
    ).to(device)
    print(f"Noise levels: {sigmas[0]:.4f} -> {sigmas[-1]:.6f}")

    # Create model
    model = get_model(config, sigmas).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        weight_decay=config['training']['weight_decay'],
    )

    # Create EMA
    ema = None
    if config['training']['use_ema']:
        ema = EMA(model, decay=config['training']['ema_rate'])

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        info = load_checkpoint(args.resume, model, optimizer, ema, device)
        start_epoch = info['epoch'] + 1
        print(f"Resumed from epoch {info['epoch']}, step {info['step']}")

    # Training loop
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train one epoch
        metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            sigmas=sigmas,
            ema=ema,
            device=device,
            epoch=epoch,
            config=config,
            grad_clip=config['training']['grad_clip'],
        )

        # Log epoch metrics
        print(f"Epoch {epoch}: Loss = {metrics['loss']:.4f}, MSE = {metrics['mse']:.4f}")

        if wandb.run is not None:
            wandb.log({
                'epoch/loss': metrics['loss'],
                'epoch/mse': metrics['mse'],
                'epoch': epoch,
            })

        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'ncsn_cifar10_epoch{epoch + 1}.pt'
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                ema=ema,
                epoch=epoch,
                step=epoch * len(train_loader),
                loss=metrics['loss'],
                config=config,
                path=checkpoint_path,
            )

        # Generate samples
        if (epoch + 1) % config['training']['sample_interval'] == 0:
            print("Generating samples...")
            samples = generate_samples(
                model=model,
                sigmas=sigmas,
                config=config,
                device=device,
                use_ema=config['training']['use_ema'],
                ema=ema,
            )

            # Make grid and log
            grid = make_grid(samples, nrow=8, normalize=True)

            if wandb.run is not None:
                wandb.log({
                    'samples': wandb.Image(grid.permute(1, 2, 0).cpu().numpy()),
                    'epoch': epoch,
                })

            # Also save locally
            import torchvision
            os.makedirs('samples', exist_ok=True)
            torchvision.utils.save_image(
                grid,
                f'samples/samples_epoch{epoch + 1}.png'
            )

    # Save final checkpoint
    final_path = os.path.join(args.checkpoint_dir, 'ncsn_cifar10_final.pt')
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        ema=ema,
        epoch=config['training']['num_epochs'] - 1,
        step=config['training']['num_epochs'] * len(train_loader),
        loss=metrics['loss'],
        config=config,
        path=final_path,
    )

    print("\nTraining complete!")

    if wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()