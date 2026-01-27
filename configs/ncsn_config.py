class NCSNConfig:
    # Data
    dataset = 'CIFAR10'
    image_size = 32
    channels = 3

    # Training
    batch_size = 128
    lr = 1e-3  # Paper uses 0.001
    n_epochs = 200  # Paper uses ~200k iterations (~500 epochs)
    num_workers = 2
    device = 'cuda'

    # NCSN specific
    sigma_start = 1.0  # Paper: sigma_1 = 1.0 (was 50.0 - this was the main bug!)
    sigma_end = 0.01   # Paper: sigma_L = 0.01
    num_scales = 10

    # Sampling / Annealed Langevin Dynamics
    n_steps_per_scale = 100  # Paper: T = 100
    step_lr = 2e-5  # Paper: epsilon = 2 x 10^-5 (was 2e-6)

    # VISUALIZATION (New)
    sample_interval = 5  # Generate images every 5 epochs
    n_samples_to_show = 16  # How many images to generate