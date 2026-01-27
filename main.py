import wandb
import torch
from configs.ncsn_config import NCSNConfig
from src.data import get_dataloader
from src.models.ncsn import ScoreNet
from src.training.train_ncsn import train_ncsn


def main():
    # 1. Init Config
    config = NCSNConfig()

    # 2. Init WandB
    # Make sure you are logged in: 'wandb login' in terminal
    wandb.init(project="cifar10-generative-models", name="ncsn-baseline", config=config.__dict__)

    # 3. Load Data
    loader = get_dataloader(config)

    # 4. Init Model
    model = ScoreNet(
        channels=config.channels,
        ch=128,
        ch_mult=[1, 2, 2, 2]
    )

    # 5. Run Training
    train_ncsn(model, loader, config)


if __name__ == "__main__":
    main()