import torch


class IWAEConfig:
    """
    Configuration for Importance Weighted Autoencoder (Burda et al., 2016)
    """
    # Data
    dataset = 'CIFAR10'
    image_size = 32
    channels = 3

    # Training
    batch_size = 64  # Reduced from 128 because k samples multiply memory usage
    lr = 1e-4
    n_epochs = 150
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # IWAE specific
    latent_dim = 128  # Same as beta-VAE for fair comparison
    n_importance_samples = 5  # k in the paper (number of importance-weighted samples)
    recon_type = 'gaussian'  # 'gaussian' for continuous data

    # Visualization
    sample_interval = 5  # Generate images every N epochs
    n_samples_to_show = 16  # Number of samples to generate
