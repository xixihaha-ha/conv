"""
CIFAR-100 dataset loader example using PyTorch torchvision

Provides:
- get_cifar100_loaders(...) -> train_loader, test_loader, classes
- Example usage in `__main__` which prints loader sizes and a sample batch shape.

Requirements:
- torch, torchvision
- Optional: matplotlib to visualize a batch

Run:
    cd f:\githubzyuu\conv
    python dataset_cifar100.py

"""

import os

import torch
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms, utils
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


def get_cifar100_loaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4,
                        download: bool = True, augment: bool = True):
    """Return train and test DataLoader for CIFAR-100.

    Args:
        data_dir: directory to download/store dataset
        batch_size: batch size passed to DataLoader
        num_workers: DataLoader num_workers
        download: whether to download dataset
        augment: whether to apply basic augmentation on train set

    Returns:
        train_loader, test_loader, classes (list of class names)
    """
    if not _HAS_TORCHVISION:
        raise RuntimeError('torchvision not available. Install with `pip install torchvision`')

    # Standard CIFAR-100 normalization (mean/std for RGB channels)
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transforms = []
    if augment:
        train_transforms += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    train_transforms += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    test_transforms = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, transform=train_transform, download=download)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, transform=test_transform, download=download)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, train_dataset.classes


def show_batch_images(batch, classes=None, denorm=None, nrow=8):
    """Optional helper to show a batch of images (requires matplotlib).

    batch: tensor of shape (N, C, H, W)
    classes: list of class names
    denorm: function to reverse normalization (tensor) -> tensor
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib not available, skipping image display')
        return

    grid = utils.make_grid(batch, nrow=nrow, normalize=False, pad_value=1)
    img = grid.permute(1, 2, 0).cpu().numpy()

    if denorm is not None:
        img = denorm(img)

    plt.figure(figsize=(10, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    print('\n=== CIFAR-100 Loader Example ===')
    if not _HAS_TORCHVISION:
        print('torchvision not found. Install with: pip install torchvision')
    else:
        # users on Windows: default num_workers=0 is safer if you get fork issues
        nw = 0 if os.name == 'nt' else 4
        train_loader, test_loader, classes = get_cifar100_loaders(data_dir='./data', batch_size=64, num_workers=nw, download=True, augment=True)

        print('Train batches:', len(train_loader), '  Test batches:', len(test_loader))

        # print a single batch shapes and labels
        batch = next(iter(train_loader))
        images, labels = batch
        print('Batch images shape:', images.shape)
        print('Batch labels shape:', labels.shape)
        print('Sample labels (first 10):', labels[:10].tolist())
        print('Class[labels[0]] =', classes[labels[0]])

        # optional: show images (will attempt matplotlib)
        try:
            # build a simple denorm for visualization
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)

            def denorm(img):
                img = img * std + mean
                img = (img - img.min()) / (img.max() - img.min())
                return img

            show_batch_images(images[:32], classes=classes, denorm=denorm, nrow=8)
        except Exception as e:
            print('Could not display images:', e)

    print('\n=== End ===')