"""
IWAE Training Entry Point
Importance Weighted Autoencoder (Burda et al., 2016)
"""
import wandb
import argparse
from configs.iwae_config import IWAEConfig
from src.data import get_dataloader
from src.models.iwae import IWAE
from src.training.train_iwae import train_iwae


def main():
    parser = argparse.ArgumentParser(description='Train IWAE on CIFAR-10')
    parser.add_argument('--k', type=int, default=None,
                        help='Number of importance samples (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Latent dimension (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    args = parser.parse_args()

    # Initialize config
    config = IWAEConfig()

    # Override config with command-line arguments
    if args.k is not None:
        config.n_importance_samples = args.k
    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.latent_dim is not None:
        config.latent_dim = args.latent_dim
    if args.lr is not None:
        config.lr = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    k = config.n_importance_samples
    print(f"=" * 50)
    print(f"IWAE Training Configuration")
    print(f"=" * 50)
    print(f"k (importance samples): {k}")
    print(f"Epochs: {config.n_epochs}")
    print(f"Latent dimension: {config.latent_dim}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Device: {config.device}")
    print(f"=" * 50)

    # Initialize WandB
    wandb.init(
        project="cifar10-generative-models",
        name=f"iwae-k{k}",
        config=config.__dict__
    )

    # Load data
    loader = get_dataloader(config)

    # Initialize model
    model = IWAE(
        in_channels=config.channels,
        latent_dim=config.latent_dim
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    train_iwae(model, loader, config)

    wandb.finish()


if __name__ == "__main__":
    main()
