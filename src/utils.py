import torch
import numpy as np

def get_sigmas(config):
    """
    Generates a geometric sequence of sigmas (noise levels).
    From sigma_start down to sigma_end.
    """
    sigmas = np.exp(np.linspace(
        np.log(config.sigma_start),
        np.log(config.sigma_end),
        config.num_scales
    ))
    return torch.tensor(sigmas).float()

def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)