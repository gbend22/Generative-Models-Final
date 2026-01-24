"""
Configuration file for NCSN training on CIFAR-10
Based on Song & Ermon (2019, 2020)
"""

import torch
import numpy as np

config = {
    # Data
    'dataset': 'cifar10',
    'data_dir': './data',
    'image_size': 32,
    'num_channels': 3,
    'num_classes': 10,  # Not used for unconditional generation

    # Model Architecture
    'model': {
        'name': 'NCSNv1',
        'ngf': 128,  # Number of generator filters (base channel dimension)
        'num_classes': 10,  # Number of noise levels
    },

    # Noise levels (geometric progression)
    # σ_1 > σ_2 > ... > σ_L
    'noise': {
        'sigma_begin': 1.0,  # Largest noise level
        'sigma_end': 0.01,  # Smallest noise level
        'num_classes': 10,  # Number of noise levels (L)
    },

    # Training
    'training': {
        'batch_size': 128,
        'n_epochs': 200,
        'n_iters': None,  # Will be computed from dataset size
        'snapshot_freq': 5000,  # Save checkpoint every N iterations
        'val_freq': 1000,  # Validate every N iterations
        'log_freq': 100,  # Log to wandb every N iterations

        # Optimization
        'optimizer': 'Adam',
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'weight_decay': 0.0,
        'amsgrad': False,

        # Learning rate scheduling
        'lr_scheduler': 'StepLR',
        'lr_decay_epochs': [150, 175],
        'lr_decay_factor': 0.1,

        # Gradient clipping
        'grad_clip': 1.0,

        # Exponential Moving Average
        'ema': True,
        'ema_decay': 0.999,
    },

    # Sampling (Annealed Langevin Dynamics)
    'sampling': {
        'n_steps_each': 100,  # Number of Langevin steps per noise level
        'step_lr': 0.00002,  # Step size for Langevin dynamics
        'final_only': True,  # Only return final samples
        'denoise': True,  # Apply one-step denoising at the end
    },

    # Evaluation
    'eval': {
        'batch_size': 256,
        'num_samples': 50000,  # For FID calculation
        'fid_stats': None,  # Path to precomputed FID stats (optional)
    },

    # Logging
    'wandb': {
        'project': 'ncsn-cifar10',
        'entity': None,  # Your wandb username
        'name': 'ncsn_baseline',
        'tags': ['ncsn', 'cifar10', 'score-matching'],
    },

    # System
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'seed': 42,

    # Paths
    'output_dir': './outputs',
    'checkpoint_dir': './checkpoints',
    'sample_dir': './samples',
}


def get_sigmas(config):
    """
    Generate geometric sequence of noise levels
    σ_i = σ_1 * (σ_L / σ_1)^((i-1)/(L-1)) for i = 1, ..., L
    """
    sigma_begin = config['noise']['sigma_begin']
    sigma_end = config['noise']['sigma_end']
    num_classes = config['noise']['num_classes']

    sigmas = np.exp(np.linspace(
        np.log(sigma_begin),
        np.log(sigma_end),
        num_classes
    ))

    return sigmas


if __name__ == '__main__':
    # Test configuration
    print("NCSN Configuration:")
    print(f"Device: {config['device']}")
    print(f"\nNoise levels:")
    sigmas = get_sigmas(config)
    for i, sigma in enumerate(sigmas):
        print(f"  σ_{i + 1} = {sigma:.4f}")
    print(f"\nTotal training iterations per epoch: ~{50000 // config['training']['batch_size']}")
    print(f"Total epochs: {config['training']['n_epochs']}")