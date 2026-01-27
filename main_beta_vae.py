import wandb
import torch
import argparse
from configs.beta_vae_config import BetaVAEConfig
from src.data import get_dataloader
from src.models.beta_vae import BetaVAE
from src.training.train_beta_vae import train_beta_vae


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Beta-VAE on CIFAR-10')
    parser.add_argument('--beta', type=float, default=None, help='Beta value (default: from config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (default: from config)')
    parser.add_argument('--latent_dim', type=int, default=None, help='Latent dimension (default: from config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default: from config)')
    args = parser.parse_args()

    # 1. Init Config
    config = BetaVAEConfig()

    # Override config with command line arguments if provided
    if args.beta is not None:
        config.beta = args.beta
    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.latent_dim is not None:
        config.latent_dim = args.latent_dim
    if args.lr is not None:
        config.lr = args.lr

    print(f"Running with beta={config.beta}, epochs={config.n_epochs}, latent_dim={config.latent_dim}")

    # 2. Init WandB
    wandb.init(
        project="cifar10-generative-models",
        name=f"beta-vae-beta{config.beta}",
        config=config.__dict__
    )

    # 3. Load Data
    loader = get_dataloader(config)

    # 4. Init Model
    model = BetaVAE(
        in_channels=config.channels,
        latent_dim=config.latent_dim
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 5. Run Training
    train_beta_vae(model, loader, config)

    # 6. Close WandB
    wandb.finish()


if __name__ == "__main__":
    main()
