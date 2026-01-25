import torch
from src.models.ncsn import NCSN
from src.samplers.langevin import annealed_langevin_sampling
from torchvision.utils import save_image

device = "cuda"
sigmas = torch.tensor([50, 25, 10, 5, 1, 0.5, 0.1, 0.01], device=device)

model = NCSN(sigmas).to(device)
model.load_state_dict(torch.load("checkpoints/ncsn_epoch_99.pt"))
model.eval()

samples = annealed_langevin_sampling(model, sigmas, device=device)
save_image((samples + 1) / 2, "samples.png", nrow=8)
