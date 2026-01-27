import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(config):
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        # Use simple 0-1 scaling or shift to mean 0. 
        # NCSN usually works fine with [0,1], but let's do simple normalization.
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        drop_last=True
    )
    return loader