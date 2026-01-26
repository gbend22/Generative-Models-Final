"""
NCSN Training Script
train.py

Usage:
    python train.py --config configs/ncsn_base.yaml
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import argparse
from tqdm import tqdm
import yaml
import numpy as np

from models.ncsn import NCSN
from losses.score_matching import DenoisingScoreMatchingLoss, AnnealedDenoisingScoreLoss
from utils.data import get_unlabeled_cifar10
from utils.visualization import save_sample_grid, visualize_noise_levels


class NCSNTrainer:
    """
    Trainer for NCSN on CIFAR-10
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = NCSN(config).to(self.device)

        # Initialize loss
        if config.get('use_annealed_loss', True):
            self.criterion = AnnealedDenoisingScoreLoss(
                self.model.sigmas,
                weighting=config.get('loss_weighting', 'exponential')
            )
        else:
            self.criterion = DenoisingScoreMatchingLoss(self.model.sigmas)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            weight_decay=config.get('weight_decay', 0)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 200),
            eta_min=config.get('lr_min', 1e-6)
        )

        # EMA for better sample quality
        if config.get('use_ema', True):
            self.ema_model = NCSN(config).to(self.device)
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_decay = config.get('ema_decay', 0.999)
        else:
            self.ema_model = None

        print(f"✓ Trainer initialized on {self.device}")
        print(f"✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def update_ema(self):
        """Update EMA model"""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        loss_dict_accum = {}

        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')

        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)

            # Forward pass
            loss, loss_dict = self.criterion(self.model, images)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # Update EMA
            self.update_ema()

            # Track losses
            total_loss += loss.item()

            # Accumulate per-level losses
            for key, val in loss_dict.items():
                if key not in loss_dict_accum:
                    loss_dict_accum[key] = []
                loss_dict_accum[key].append(val)

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log to wandb
            if batch_idx % self.config.get('log_interval', 100) == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': epoch * len(dataloader) + batch_idx
                })

        # Average losses
        avg_loss = total_loss / len(dataloader)

        # Average per-level losses
        avg_loss_dict = {
            key: np.mean(vals) for key, vals in loss_dict_accum.items()
        }

        # Log epoch summary
        wandb.log({
            'train/epoch_loss': avg_loss,
            **{f'train/{k}': v for k, v in avg_loss_dict.items()},
            'epoch': epoch
        })

        return avg_loss

    @torch.no_grad()
    def sample_images(self, n_samples=64):
        """Generate sample images"""
        model = self.ema_model if self.ema_model else self.model
        model.eval()

        samples = model.sample(
            batch_size=n_samples,
            device=self.device,
            n_steps_each=self.config.get('sample_steps', 100),
            step_lr=self.config.get('sample_step_lr', 2e-5)
        )

        return samples

    def save_checkpoint(self, epoch, save_dir):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }

        if self.ema_model:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()

        path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.ema_model and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])

        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch']


def train_ncsn(config):
    """Main training function"""

    # Initialize wandb
    wandb.init(
        project=config.get('project_name', 'cifar10-ncsn'),
        name=config.get('run_name', 'ncsn-baseline'),
        config=config
    )

    # Setup data
    train_loader = get_unlabeled_cifar10(
        batch_size=config.get('batch_size', 128),
        num_workers=config.get('num_workers', 4)
    )

    # Initialize trainer
    trainer = NCSNTrainer(config)

    # Training loop
    start_epoch = 0
    if config.get('resume_checkpoint'):
        start_epoch = trainer.load_checkpoint(config['resume_checkpoint']) + 1

    for epoch in range(start_epoch, config.get('epochs', 200)):
        # Train
        avg_loss = trainer.train_epoch(train_loader, epoch)

        # Update learning rate
        trainer.scheduler.step()

        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")

        # Generate samples
        if (epoch + 1) % config.get('sample_interval', 10) == 0:
            print("  Generating samples...")
            samples = trainer.sample_images(n_samples=64)

            # Save sample grid
            save_path = f"samples/epoch_{epoch + 1}.png"
            save_sample_grid(samples, save_path, nrow=8)

            # Log to wandb
            wandb.log({
                'samples': wandb.Image(save_path),
                'epoch': epoch
            })

        # Save checkpoint
        if (epoch + 1) % config.get('save_interval', 20) == 0:
            trainer.save_checkpoint(epoch + 1, config.get('checkpoint_dir', 'checkpoints'))

    # Save final checkpoint
    trainer.save_checkpoint(config['epochs'], config.get('checkpoint_dir', 'checkpoints'))

    wandb.finish()
    print("\n✓ Training complete!")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NCSN on CIFAR-10')
    parser.add_argument('--config', type=str, default='configs/ncsn_base.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # Default config if file doesn't exist
        config = {
            # Model
            'in_channels': 3,
            'base_channels': 128,
            'num_noise_levels': 10,
            'sigma_begin': 1.0,
            'sigma_end': 0.01,

            # Training
            'epochs': 200,
            'batch_size': 128,
            'lr': 1e-4,
            'lr_min': 1e-6,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0,
            'grad_clip': 1.0,

            # Loss
            'use_annealed_loss': True,
            'loss_weighting': 'exponential',

            # EMA
            'use_ema': True,
            'ema_decay': 0.999,

            # Sampling
            'sample_steps': 100,
            'sample_step_lr': 2e-5,

            # Logging
            'project_name': 'cifar10-ncsn',
            'run_name': 'ncsn-baseline',
            'log_interval': 100,
            'sample_interval': 10,
            'save_interval': 20,
            'checkpoint_dir': 'checkpoints',

            # Data
            'num_workers': 4,
        }

    if args.resume:
        config['resume_checkpoint'] = args.resume

    # Train
    train_ncsn(config)