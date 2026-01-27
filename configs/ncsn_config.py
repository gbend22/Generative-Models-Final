class NCSNConfig:
    # Data
    dataset = 'CIFAR10'
    image_size = 32
    channels = 3

    # Training
    batch_size = 128
    lr = 1e-4
    n_epochs = 50
    num_workers = 2
    device = 'cuda'

    # NCSN specific
    sigma_start = 50.0
    sigma_end = 0.01
    num_scales = 10

    # Sampling / Annealed Langevin Dynamics
    n_steps_per_scale = 100
    step_lr = 2e-6

    # VISUALIZATION (New)
    sample_interval = 5  # Generate images every 5 epochs
    n_samples_to_show = 16  # How many images to generate