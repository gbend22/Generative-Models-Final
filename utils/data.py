"""
CIFAR-10 Data Loading and Preprocessing
utils/data.py
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_cifar10_dataloaders(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Get CIFAR-10 train and test dataloaders

    Args:
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        data_dir: Directory to store CIFAR-10 data

    Returns:
        train_loader, test_loader
    """

    # Data augmentation for training (optional, for score models usually minimal)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
    ])

    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"✓ CIFAR-10 loaded:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")

    return train_loader, test_loader


def get_data_statistics(dataloader):
    """
    Calculate mean and std of dataset (for verification)
    """
    mean = 0.
    std = 0.
    total_samples = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean, std


class UnlabeledCIFAR10(torch.utils.data.Dataset):
    """
    CIFAR-10 dataset without labels (for unconditional generation)
    """
    def __init__(self, root='./data', train=True, transform=None):
        self.cifar = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        image, _ = self.cifar[idx]  # Ignore label
        return image


def get_unlabeled_cifar10(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Get CIFAR-10 without labels (cleaner for unconditional generation)
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = UnlabeledCIFAR10(
        root=data_dir,
        train=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return train_loader


# Test the data loading
if __name__ == "__main__":
    print("Testing CIFAR-10 data loading...")
    print("=" * 60)

    # Get dataloaders
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)

    # Get one batch
    images, labels = next(iter(train_loader))

    print(f"\nBatch info:")
    print(f"  Image shape: {images.shape}")
    print(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {labels.unique().tolist()}")

    # Test unlabeled version
    print("\n" + "=" * 60)
    print("Testing unlabeled CIFAR-10...")
    unlabeled_loader = get_unlabeled_cifar10(batch_size=128)

    images = next(iter(unlabeled_loader))
    print(f"\nUnlabeled batch shape: {images.shape}")

    print("\n✓ Data loading working correctly!")