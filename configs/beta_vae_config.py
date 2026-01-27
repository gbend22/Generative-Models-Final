class BetaVAEConfig:
    # Data
    dataset = 'CIFAR10'
    image_size = 32
    channels = 3

    # Training
    batch_size = 128
    lr = 1e-4  # Paper uses 1e-4 for Adam
    n_epochs = 100
    num_workers = 2
    device = 'cuda'

    # Beta-VAE specific
    latent_dim = 32  # Paper uses 32 for CelebA/CIFAR-like datasets
    beta = 4.0  # Key hyperparameter: beta > 1 for disentanglement
    # Paper uses beta=4 for simple datasets, beta=250 for CelebA
    # For CIFAR-10, start with beta=4 and tune

    # Reconstruction type
    # 'gaussian' for continuous data, 'bernoulli' for binary
    recon_type = 'gaussian'

    # Visualization
    sample_interval = 5  # Generate images every N epochs
    n_samples_to_show = 16  # Number of samples to generate
