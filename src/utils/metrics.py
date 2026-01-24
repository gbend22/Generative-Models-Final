"""
Evaluation metrics for generative models
- Fréchet Inception Distance (FID)
- Inception Score (IS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from tqdm import tqdm
import torchvision.models as models


class InceptionV3(nn.Module):
    """
    Pretrained InceptionV3 network for feature extraction
    Returns pool3 features (2048-dim) for FID and logits for IS
    """

    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()

        # Remove final layers
        self.features = nn.Sequential(*list(inception.children())[:-1])
        self.fc = inception.fc

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: (B, 3, 299, 299) images in range [0, 1]
        Returns:
            features: (B, 2048) pool3 features
            logits: (B, 1000) classification logits
        """
        # Inception expects [0,1] images, resize to 299x299
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Get features before final pooling
        features = self.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)

        # Get logits
        logits = self.fc(features)

        return features, logits


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet Distance between two Gaussians

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

    Args:
        mu1: mean of first distribution
        sigma1: covariance of first distribution
        mu2: mean of second distribution
        sigma2: covariance of second distribution

    Returns:
        Fréchet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


@torch.no_grad()
def calculate_activation_statistics(images, model, batch_size=50, device='cuda'):
    """
    Calculate mean and covariance of Inception features

    Args:
        images: (N, C, H, W) tensor of images in range [-1, 1]
        model: InceptionV3 model
        batch_size: batch size for processing
        device: torch device

    Returns:
        mu: mean of features
        sigma: covariance of features
    """
    model.eval()
    model.to(device)

    # Convert images from [-1, 1] to [0, 1]
    images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0, 1)

    n_samples = images.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Collect features
    features_list = []

    for i in tqdm(range(n_batches), desc='Extracting features'):
        start = i * batch_size
        end = min(start + batch_size, n_samples)
        batch = images[start:end].to(device)

        features, _ = model(batch)
        features_list.append(features.cpu().numpy())

    features = np.concatenate(features_list, axis=0)

    # Calculate statistics
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    return mu, sigma


@torch.no_grad()
def calculate_fid(real_images, fake_images, batch_size=50, device='cuda'):
    """
    Calculate Fréchet Inception Distance

    Args:
        real_images: (N, 3, 32, 32) real images in [-1, 1]
        fake_images: (N, 3, 32, 32) generated images in [-1, 1]
        batch_size: batch size for processing
        device: torch device

    Returns:
        FID score
    """
    model = InceptionV3().to(device)

    # Calculate statistics for real images
    print("Calculating statistics for real images...")
    mu_real, sigma_real = calculate_activation_statistics(real_images, model, batch_size, device)

    # Calculate statistics for fake images
    print("Calculating statistics for generated images...")
    mu_fake, sigma_fake = calculate_activation_statistics(fake_images, model, batch_size, device)

    # Calculate FID
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    return fid_score


@torch.no_grad()
def calculate_inception_score(images, model, batch_size=50, splits=10, device='cuda'):
    """
    Calculate Inception Score

    IS = exp(E_x[KL(p(y|x) || p(y))])

    Args:
        images: (N, 3, 32, 32) images in [-1, 1]
        model: InceptionV3 model
        batch_size: batch size for processing
        splits: number of splits for computing mean and std
        device: torch device

    Returns:
        mean IS, std IS
    """
    model.eval()
    model.to(device)

    # Convert images from [-1, 1] to [0, 1]
    images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0, 1)

    n_samples = images.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Collect predictions
    preds = []

    for i in tqdm(range(n_batches), desc='Computing predictions'):
        start = i * batch_size
        end = min(start + batch_size, n_samples)
        batch = images[start:end].to(device)

        _, logits = model(batch)
        preds.append(F.softmax(logits, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Calculate IS for each split
    scores = []
    split_size = n_samples // splits

    for i in range(splits):
        start = i * split_size
        end = (i + 1) * split_size if i < splits - 1 else n_samples
        part = preds[start:end]

        # p(y|x)
        py_x = part
        # p(y) = E_x[p(y|x)]
        py = np.mean(part, axis=0, keepdims=True)

        # KL divergence
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.mean(np.sum(kl, axis=1))

        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


class FIDCalculator:
    """
    Helper class for FID calculation with caching
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.model = InceptionV3().to(device)
        self.real_stats = None

    def compute_real_stats(self, real_images, batch_size=50):
        """Compute and cache real image statistics"""
        self.real_stats = calculate_activation_statistics(
            real_images, self.model, batch_size, self.device
        )

    def compute_fid(self, fake_images, batch_size=50):
        """Compute FID using cached real statistics"""
        if self.real_stats is None:
            raise ValueError("Must compute real statistics first")

        mu_real, sigma_real = self.real_stats
        mu_fake, sigma_fake = calculate_activation_statistics(
            fake_images, self.model, batch_size, self.device
        )

        return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


if __name__ == '__main__':
    # Test metrics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Creating dummy images...")
    real_images = torch.randn(100, 3, 32, 32)
    fake_images = torch.randn(100, 3, 32, 32)

    print("\nCalculating FID...")
    fid = calculate_fid(real_images, fake_images, batch_size=10, device=device)
    print(f"FID: {fid:.2f}")

    print("\nCalculating Inception Score...")
    is_mean, is_std = calculate_inception_score(fake_images, InceptionV3(), batch_size=10, device=device)
    print(f"IS: {is_mean:.2f} ± {is_std:.2f}")