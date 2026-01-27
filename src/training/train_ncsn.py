import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import os
import torchvision

from src.training.loss import denoising_score_matching_loss
from src.utils import get_sigmas, save_checkpoint
from src.training.sampling import annealed_langevin_dynamics  # Import sampling logic


def train_ncsn(model, dataloader, config, plot_callback=None):
    """
    Args:
        model: The ScoreNet
        dataloader: Training data
        config: Configuration object
        plot_callback: Optional function to display images in Colab (func(images, epoch))
    """
    # 1. Setup
    device = config.device
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    sigmas = get_sigmas(config).to(device)

    print(f"Starting training on {device}...")

    # 2. Epoch Loop
    for epoch in range(config.n_epochs):
        model.train()  # Ensure model is in train mode
        avg_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.n_epochs}")

        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device)

            optimizer.zero_grad()
            loss = denoising_score_matching_loss(model, x, sigmas)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            wandb.log({"train_loss": loss.item()})

        # End of Epoch Stats
        avg_loss /= len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.5f}")

        # --- SAMPLING & VISUALIZATION STEP ---
        if (epoch + 1) % config.sample_interval == 0:
            print(f"Epoch {epoch + 1}: Running Sampling...")

            # Generate samples (Model handles eval/train mode inside, but let's be safe)
            samples = annealed_langevin_dynamics(
                model, sigmas, config, n_samples=config.n_samples_to_show
            )

            # 1. Log to WandB
            # Create a grid for WandB
            grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)
            wandb.log({
                "generated_images": [wandb.Image(grid, caption=f"Epoch {epoch + 1}")]
            })

            # 2. Show in Colab (via callback)
            if plot_callback:
                plot_callback(samples, epoch + 1)

            # Set model back to train mode for next epoch
            model.train()
        # -------------------------------------

        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_checkpoint(model, optimizer, epoch, path=f"checkpoints/ncsn_epoch_{epoch + 1}.pth")

    print("Training Complete.")