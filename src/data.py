"""
data.py
=======
CIFAR-10 loading, preprocessing and augmentation.

Both models MUST see exactly the same data pipeline for the comparison to be
fair, so this module is the single source of truth used by both training
runs in main.py.
"""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

# Standard CIFAR-10 per-channel mean/std (widely used, computed over the
# training set). Used for normalization for both models.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(augment=True):
    """Train transform (with augmentation) and eval transform (no augmentation)."""
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    if augment:
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),   # standard CIFAR augmentation
            T.RandomHorizontalFlip(),      # standard CIFAR augmentation
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_tf = eval_tf

    return train_tf, eval_tf


def get_dataloaders(data_dir="./data", batch_size=128, val_fraction=0.1,
                     augment=True, num_workers=2, seed=42):
    """
    Returns train_loader, val_loader, test_loader.

    A held-out validation split is carved out of the official 50,000-image
    training set (val_fraction of it) so that the official 10,000-image
    test set stays untouched until final evaluation, exactly as the
    assignment expects ("training and validation curves" + "final test
    accuracy").
    """
    train_tf, eval_tf = get_transforms(augment=augment)

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf)
    # A second copy with eval-time transforms, used only for the validation
    # split so validation images are NOT augmented.
    full_train_eval_tf = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=eval_tf)

    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(
        range(len(full_train)), [n_train, n_val], generator=generator)

    train_set = torch.utils.data.Subset(full_train, train_idx.indices)
    val_set = torch.utils.data.Subset(full_train_eval_tf, val_idx.indices)

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
