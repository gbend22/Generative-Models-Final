import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloader(
    data_dir="./data",
    batch_size=128,
    num_workers=4,
    train=True
):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train
    )
