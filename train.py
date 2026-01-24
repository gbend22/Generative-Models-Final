"""
Training script for NCSN on CIFAR-10
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb

from configs.ncsn_cifar10 import config
from src.models.NCSN import NCSN, ExponentialMovingAverage
from src.losses.score_matching import get_loss_fn
from src.datasets.cifar10 import get_cifar10_dataloaders
from src.utils.visualization import save_image_grid, visualize_samples
from src.samplers.langevin import get_sampler


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_optimizer(model, config):
    """Create optimizer"""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        eps=config['training']['eps'],
        weight_decay=config['training']['weight_decay'],
        amsgrad=config['training']['amsgrad']
    )
    return optimizer


def get_scheduler(optimizer, config):
    """Create learning rate scheduler"""
    if config['training']['lr_scheduler'] == 'StepLR':
        # Multi-step learning rate decay
        milestones = config['training']['lr_decay_epochs']
        gamma = config['training']['lr_decay_factor']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    else:
        scheduler = None

    return scheduler


def train(args):
    """Main training function"""

    # Set seed
    set_seed(config['seed'])

    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['wandb']['name'],
            config=config,
            tags=config['wandb']['tags']
        )

    # Setup device
    device = torch.device(config['device'])
    print(f"Using device: {device}")

    # Create dataloaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(config, augment=True)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Create model
    print("Creating model...")
    model = NCSN(config).to(device)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Create EMA
    if config['training']['ema']:
        ema = ExponentialMovingAverage(model, decay=config['training']['ema_decay'])
    else:
        ema = None

    # Create loss function
    loss_fn = get_loss_fn(config).to(device)

    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Training loop
    print("\nStarting training...")
    global_step = 0
    best_loss = float('inf')

    for epoch in range(config['training']['n_epochs']):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["training"]["n_epochs"]}')

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)

            # Forward pass
            loss, loss_dict = loss_fn(model, images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['grad_clip']
                )

            optimizer.step()

            # Update EMA
            if ema is not None:
                ema.update()

            # Logging
            epoch_losses.append(loss.item())
            global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log to wandb
            if not args.no_wandb and global_step % config['training']['log_freq'] == 0:
                log_dict = {
                    'train/loss': loss.item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                }
                wandb.log(log_dict, step=global_step)

            # Generate samples
            if global_step % config['training']['val_freq'] == 0:
                print(f"\nGenerating samples at step {global_step}...")
                model.eval()

                # Use EMA model if available
                if ema is not None:
                    ema.apply_shadow()

                with torch.no_grad():
                    sampler = get_sampler(config, model.sigmas.cpu().numpy())
                    samples = sampler.sample(
                        model,
                        batch_size=64,
                        device=device,
                        verbose=False
                    )

                    # Save samples
                    save_path = os.path.join(
                        config['sample_dir'],
                        f'samples_step_{global_step}.png'
                    )
                    visualize_samples(samples, save_path, nrow=8)

                    if not args.no_wandb:
                        wandb.log({
                            'samples': wandb.Image(save_path)
                        }, step=global_step)

                # Restore original parameters
                if ema is not None:
                    ema.restore()

                model.train()

            # Save checkpoint
            if global_step % config['training']['snapshot_freq'] == 0:
                checkpoint_path = os.path.join(
                    config['checkpoint_dir'],
                    f'checkpoint_step_{global_step}.pth'
                )

                checkpoint = {
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                }

                if ema is not None:
                    checkpoint['ema_state_dict'] = ema.state_dict()

                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()

                torch.save(checkpoint, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")

        # Epoch complete
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(config['checkpoint_dir'], 'model_best.pth')

            checkpoint = {
                'step': global_step,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': config,
            }

            if ema is not None:
                checkpoint['ema_state_dict'] = ema.state_dict()

            torch.save(checkpoint, best_path)
            print(f"Saved best model with loss {best_loss:.4f}")

    print("\nTraining complete!")

    # Save final model
    final_path = os.path.join(config['checkpoint_dir'], 'model_final.pth')
    checkpoint = {
        'step': global_step,
        'epoch': config['training']['n_epochs'],
        'model_state_dict': model.state_dict(),
        'config': config,
    }

    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()

    torch.save(checkpoint, final_path)
    print(f"Saved final model to {final_path}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NCSN on CIFAR-10')
    parser.add_argument('--config', type=str, default='configs/ncsn_cifar10.py',
                        help='Path to config file')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    train(args)