import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import os
import torchvision

from src.training.loss_beta_vae import beta_vae_loss
from src.utils import save_checkpoint


def train_beta_vae(model, dataloader, config, plot_callback=None):
    """
    Training loop for Beta-VAE.

    Args:
        model: BetaVAE model
        dataloader: Training data loader
        config: Configuration object
        plot_callback: Optional callback for visualization
    """
    device = config.device
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    print(f"Starting Beta-VAE training on {device}...")
    print(f"Beta value: {config.beta}")
    print(f"Latent dimension: {config.latent_dim}")

    for epoch in range(config.n_epochs):
        model.train()
        total_loss_epoch = 0.0
        recon_loss_epoch = 0.0
        kl_loss_epoch = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.n_epochs}")

        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device)

            # Forward pass
            optimizer.zero_grad()
            x_recon, mu, log_var = model(x)

            # Compute loss
            total_loss, recon_loss, kl_loss = beta_vae_loss(
                x_recon, x, mu, log_var,
                beta=config.beta,
                recon_type=config.recon_type
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss_epoch += total_loss.item()
            recon_loss_epoch += recon_loss.item()
            kl_loss_epoch += kl_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item()
            })

            # Log to WandB
            wandb.log({
                "train_loss": total_loss.item(),
                "recon_loss": recon_loss.item(),
                "kl_loss": kl_loss.item(),
                "beta_kl_loss": config.beta * kl_loss.item()
            })

        # End of epoch stats
        n_batches = len(dataloader)
        avg_total = total_loss_epoch / n_batches
        avg_recon = recon_loss_epoch / n_batches
        avg_kl = kl_loss_epoch / n_batches

        print(f"Epoch {epoch + 1} - Loss: {avg_total:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

        # Sampling and visualization
        if (epoch + 1) % config.sample_interval == 0:
            print(f"Epoch {epoch + 1}: Generating samples...")

            model.eval()
            with torch.no_grad():
                # Generate samples from prior
                samples = model.sample(config.n_samples_to_show, device)

                # Get some reconstructions
                x_sample = next(iter(dataloader))[0][:config.n_samples_to_show].to(device)
                x_recon_sample, _, _ = model(x_sample)

            # Create grids
            samples_grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)
            recon_grid = torchvision.utils.make_grid(
                torch.cat([x_sample, x_recon_sample], dim=0),
                nrow=config.n_samples_to_show,
                padding=2
            )

            # Log to WandB
            wandb.log({
                "generated_samples": [wandb.Image(samples_grid, caption=f"Epoch {epoch + 1}")],
                "reconstructions": [wandb.Image(recon_grid, caption=f"Epoch {epoch + 1} (top: original, bottom: recon)")]
            })

            # Callback for Colab visualization
            if plot_callback:
                plot_callback(samples, epoch + 1)

            model.train()

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_checkpoint(
                model, optimizer, epoch,
                path=f"checkpoints/beta_vae_epoch_{epoch + 1}.pth"
            )

    print("Training Complete.")
