import wandb
import torch
from configs.beta_vae_config import BetaVAEConfig
from src.data import get_dataloader
from src.models.beta_vae import BetaVAE
from src.training.train_beta_vae import train_beta_vae


def main():
    # 1. Init Config
    config = BetaVAEConfig()

    # 2. Init WandB
    # Make sure you are logged in: 'wandb login' in terminal
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
