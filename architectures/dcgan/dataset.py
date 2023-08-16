import torch
import config
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from typing import Tuple


def create_dataloders() -> Tuple[DataLoader, DataLoader]:
    """Creates dataloaders for MNIST dataset.

    Returns:
        Tuple[DataLoader, DataLoader]: Train DataLoader and Test DataLoader.
    """

    # Download MNIST dataset
    custom_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    train_dataset = MNIST(
        root="./data/mnist/train",
        download=True,
        train=True,
        transform=custom_transform,
    )

    test_dataset = MNIST(
        root="./data/mnist/test", download=True, train=False, transform=custom_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, test_loader
