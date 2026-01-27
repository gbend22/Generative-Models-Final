"""
Data loading utilities for CIFAR-10.

Handles:
- Dataset loading with proper normalization
- Data augmentation (optional)
- DataLoader creation

CIFAR-10 images are 32x32 RGB images normalized to [-1, 1] range.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional


def get_cifar10_transforms(
        train: bool = True,
        augment: bool = True,
        image_size: int = 32,
) -> transforms.Compose:
    """
    Get transforms for CIFAR-10.

    Training transforms can include:
    - Random horizontal flip
    - Random crop with padding

    All transforms normalize to [-1, 1] range.

    Args:
        train: Whether this is for training data
        augment: Whether to apply data augmentation
        image_size: Target image size (should be 32 for CIFAR-10)

    Returns:
        Composed transforms
    """
    transform_list = []

    # Data augmentation for training
    if train and augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            # Optional: random crop with padding
            # transforms.RandomCrop(image_size, padding=4),
        ])

    # Convert to tensor and normalize to [-1, 1]
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),  # This maps [0, 1] -> [-1, 1]
    ])

    return transforms.Compose(transform_list)


def get_cifar10_dataloader(
        root: str = './data',
        train: bool = True,
        batch_size: int = 128,
        num_workers: int = 4,
        augment: bool = True,
        shuffle: Optional[bool] = None,
        drop_last: bool = True,
) -> DataLoader:
    """
    Create CIFAR-10 DataLoader.

    Note: CIFAR-10 has 50,000 training images and 10,000 test images.
    Class labels are NOT used for unconditional generation.

    Args:
        root: Root directory for data storage
        train: Load training or test set
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Apply data augmentation
        shuffle: Shuffle data (default: True for train, False for test)
        drop_last: Drop last incomplete batch

    Returns:
        DataLoader for CIFAR-10
    """
    if shuffle is None:
        shuffle = train

    transform = get_cifar10_transforms(train=train, augment=augment)

    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    return dataloader


def get_test_dataloader(
        root: str = './data',
        batch_size: int = 128,
        num_workers: int = 4,
) -> DataLoader:
    """Convenience function for test dataloader."""
    return get_cifar10_dataloader(
        root=root,
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
        shuffle=False,
        drop_last=False,
    )


class InfiniteDataLoader:
    """
    Infinite data loader that cycles through the dataset.

    Useful for iteration-based training rather than epoch-based.
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch


def compute_data_statistics(
        dataloader: DataLoader,
) -> dict:
    """
    Compute statistics of the dataset.

    Useful for setting noise schedule parameters.

    Args:
        dataloader: DataLoader for the dataset

    Returns:
        Dict with mean, std, min, max values
    """
    all_data = []

    for batch, _ in dataloader:
        all_data.append(batch)
        if len(all_data) > 10:  # Sample subset
            break

    data = torch.cat(all_data, dim=0)

    return {
        'mean': data.mean().item(),
        'std': data.std().item(),
        'min': data.min().item(),
        'max': data.max().item(),
        'shape': tuple(data.shape[1:]),
        'num_samples': len(dataloader.dataset),
    }


# For quick testing
if __name__ == '__main__':
    # Test data loading
    train_loader = get_cifar10_dataloader(batch_size=32)

    batch, labels = next(iter(train_loader))
    print(f"Batch shape: {batch.shape}")
    print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")

    stats = compute_data_statistics(train_loader)
    print(f"Dataset statistics: {stats}")