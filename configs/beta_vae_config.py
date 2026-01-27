class BetaVAEConfig:
    # Data
    dataset = 'CIFAR10'
    image_size = 32
    channels = 3

    # Training
    batch_size = 128
    lr = 1e-4  # Paper uses 1e-4 for Adam
    n_epochs = 150
    num_workers = 2
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

    # Beta-VAE specific
    latent_dim = 128  # Increased for CIFAR-10 complexity
    beta = 1.0  # Start with standard VAE (beta=1), then increase for disentanglement
    # Paper uses beta=4 for simple datasets - but CIFAR-10 needs more capacity first

    # Reconstruction type
    # 'gaussian' for continuous data, 'bernoulli' for binary
    recon_type = 'gaussian'

    # Visualization
    sample_interval = 5  # Generate images every N epochs
    n_samples_to_show = 16  # Number of samples to generate
