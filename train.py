import torch
from src.models.ncsn import NCSN
from src.datasets.cifar10 import get_cifar10_dataloader
from src.losses.score_matching import denoising_score_matching_loss

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sigmas = torch.tensor(
        [50, 25, 10, 5, 1, 0.5, 0.1, 0.01],
        device=device
    )

    model = NCSN(sigmas).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    loader = get_cifar10_dataloader(batch_size=128)

    for epoch in range(100):
        for x, _ in loader:
            x = x.to(device)
            loss = denoising_score_matching_loss(model, x, sigmas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} | Loss {loss.item():.4f}")
        torch.save(model.state_dict(), f"checkpoints/ncsn_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()
