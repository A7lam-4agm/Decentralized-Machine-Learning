import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_cifar10_data(batch_size=64, validation_split=0.2, subset_size=None):
    """
    Load CIFAR-10 dataset with optional subset for faster experimentation
    """

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load training and test datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create a subset if specified (for faster experimentation)
    if subset_size is not None:
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, subset_indices)

    # Split training data into train and validation
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, 10  # 10 classes in CIFAR-10