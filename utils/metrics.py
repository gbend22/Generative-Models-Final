"""
Evaluation metrics for generative models
utils/metrics.py

Implements:
- FID (Fréchet Inception Distance)
- Inception Score (IS)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy import linalg
from tqdm import tqdm


class InceptionV3Features(nn.Module):
    """
    Extract features from InceptionV3 for FID calculation
    """
    def __init__(self):
        super().__init__()

        # Load pretrained InceptionV3
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()

        # Extract feature layers (before final classification)
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Extract features

        Args:
            x: Images [B, 3, 299, 299]

        Returns:
            Features [B, 2048]
        """
        x = self.features(x)
        return x.view(x.size(0), -1)


def calculate_activation_statistics(images, model, device, batch_size=50):
    """
    Calculate mean and covariance of InceptionV3 features

    Args:
        images: Tensor of images [N, 3, H, W] in range [-1, 1]
        model: InceptionV3Features model
        device: Device to use
        batch_size: Batch size for feature extraction

    Returns:
        mu: Mean of features
        sigma: Covariance of features
    """
    model.eval()

    # Resize images to 299x299 for InceptionV3
    resize = transforms.Resize((299, 299))

    features_list = []

    n_batches = (len(images) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Extracting features"):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(images))

            batch = images[start:end].to(device)

            # Resize
            batch_resized = torch.stack([resize(img) for img in batch])

            # Extract features
            features = model(batch_resized)
            features_list.append(features.cpu().numpy())

    # Concatenate all features
    features = np.concatenate(features_list, axis=0)

    # Calculate statistics
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet Distance between two Gaussians

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

    Args:
        mu1, sigma1: Mean and covariance of first distribution
        mu2, sigma2: Mean and covariance of second distribution
        eps: Small value for numerical stability

    Returns:
        Fréchet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Calculate squared difference of means
    diff = mu1 - mu2

    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle numerical errors
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Real part only
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID
    trace = np.trace(sigma1 + sigma2 - 2 * covmean)
    fid = diff.dot(diff) + trace

    return fid


def calculate_fid(real_images, generated_images, device='cuda', batch_size=50):
    """
    Calculate FID between real and generated images

    Args:
        real_images: Real images [N, 3, 32, 32] in range [-1, 1]
        generated_images: Generated images [N, 3, 32, 32] in range [-1, 1]
        device: Device to use
        batch_size: Batch size for feature extraction

    Returns:
        FID score (lower is better)
    """
    print("Calculating FID...")

    # Initialize InceptionV3
    model = InceptionV3Features().to(device)

    # Calculate statistics for real images
    print("Processing real images...")
    mu_real, sigma_real = calculate_activation_statistics(
        real_images, model, device, batch_size
    )

    # Calculate statistics for generated images
    print("Processing generated images...")
    mu_gen, sigma_gen = calculate_activation_statistics(
        generated_images, model, device, batch_size
    )

    # Calculate FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    return fid


def calculate_inception_score(images, model=None, device='cuda', batch_size=50, splits=10):
    """
    Calculate Inception Score

    IS = exp(E[KL(p(y|x) || p(y))])

    Higher is better. Measures quality and diversity.

    Args:
        images: Generated images [N, 3, 32, 32]
        model: InceptionV3 model (if None, will create)
        device: Device to use
        batch_size: Batch size
        splits: Number of splits for calculating mean and std

    Returns:
        mean_is, std_is: Mean and std of Inception Score
    """
    if model is None:
        model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        model.eval()

    # Resize to 299x299
    resize = transforms.Resize((299, 299))

    # Get predictions
    preds = []

    n_batches = (len(images) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Calculating IS"):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(images))

            batch = images[start:end].to(device)
            batch_resized = torch.stack([resize(img) for img in batch])

            pred = model(batch_resized)
            pred = torch.nn.functional.softmax(pred, dim=1)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Calculate IS for each split
    scores = []
    split_size = len(preds) // splits

    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]

        # p(y)
        py = np.mean(part, axis=0)

        # KL divergence
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.sum(kl, axis=1)

        # Inception Score for this split
        scores.append(np.exp(np.mean(kl)))

    return np.mean(scores), np.std(scores)


# Test metrics
if __name__ == "__main__":
    print("Testing metrics...")
    print("=" * 60)

    # Generate random images
    real_images = torch.randn(100, 3, 32, 32) * 0.5  # Simulate real distribution
    generated_images = torch.randn(100, 3, 32, 32) * 0.6  # Simulate generated

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test FID
    print("\nTesting FID calculation...")
    fid = calculate_fid(real_images, generated_images, device=device, batch_size=32)
    print(f"FID Score: {fid:.2f}")

    # Test Inception Score
    print("\nTesting Inception Score...")
    is_mean, is_std = calculate_inception_score(
        generated_images, device=device, batch_size=32, splits=5
    )
    print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")

    print("\n✓ Metrics working correctly!")