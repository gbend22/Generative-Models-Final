"""
CIFAR-10 dataset utilities
Handles loading and preprocessing
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_cifar10_transforms(augment=True):
    """
    Get data transforms for CIFAR-10

    Args:
        augment: if True, apply data augmentation

    Returns:
        transform for training/testing
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
        ])
    else:
        # Testing transforms (no augmentation)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    return transform


def get_cifar10_dataloaders(config, augment=True):
    """
    Create CIFAR-10 dataloaders

    Args:
        config: experiment configuration
        augment: if True, apply data augmentation to training set

    Returns:
        train_loader, test_loader
    """
    train_transform = get_cifar10_transforms(augment=augment)
    test_transform = get_cifar10_transforms(augment=False)

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=config['data_dir'],
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=config['data_dir'],
        train=False,
        download=True,
        transform=test_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable training
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    return train_loader, test_loader


def get_cifar10_statistics():
    """
    Get mean and std of CIFAR-10 dataset
    Useful for normalization
    """
    return {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2435, 0.2616)
    }


def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1]
    """
    return (tensor + 1.0) / 2.0


def normalize(tensor):
    """
    Normalize tensor from [0, 1] to [-1, 1]
    """
    return tensor * 2.0 - 1.0


class InfiniteDataLoader:
    """
    Wrapper around DataLoader that loops infinitely
    Useful for training with iteration-based rather than epoch-based loops
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch


def get_infinite_dataloader(config):
    """
    Create infinite dataloader for iteration-based training
    """
    train_loader, _ = get_cifar10_dataloaders(config)
    return InfiniteDataLoader(train_loader)


if __name__ == '__main__':
    # Test dataset loading
    from configs.ncsn_cifar10 import config

    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(config)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Check a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Unique labels: {torch.unique(labels).tolist()}")

    # Test denormalization
    denorm_images = denormalize(images)
    print(f"\nDenormalized range: [{denorm_images.min():.3f}, {denorm_images.max():.3f}]")

    # Test infinite dataloader
    print("\nTesting infinite dataloader...")
    infinite_loader = get_infinite_dataloader(config)
    for i in range(3):
        batch, _ = next(infinite_loader)
        print(f"Batch {i + 1} shape: {batch.shape}")