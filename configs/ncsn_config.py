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

    # --- CRITICAL FIXES BELOW ---

    # NCSN specific
    # 50.0 was too high for [0,1] data. 1.0 is standard for CIFAR-10.
    sigma_start = 1.0
    sigma_end = 0.01
    num_scales = 10

    # Sampling
    n_steps_per_scale = 100

    # We increase this slightly because our sigma ratio is now smaller
    step_lr = 1e-5

    # VISUALIZATION
    sample_interval = 5
    n_samples_to_show = 16