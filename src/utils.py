import copy
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

class EMAHelper:
    """Exponential Moving Average of model parameters for stable sampling."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)