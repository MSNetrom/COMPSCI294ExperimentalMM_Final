import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from typing import Tuple, Dict

def load_cifar10() -> Tuple[Dataset, Dataset]:

    # Load the CIFAR10 dataset
    cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.PILToTensor())
    cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.PILToTensor())

    return cifar_train_dataset, cifar_test_dataset

def prepare_data_loaders(batch_size=50, num_workers=4) -> Tuple[Dict[str, DataLoader], Dict[str, Dataset], transforms.Normalize]:

    # Prepare dataset
    cifar_train_dataset, cifar_test_dataset = load_cifar10()

    # Create a normalizer for better training
    d_mean = cifar_train_dataset.data.mean(axis=(0,1,2)) / 255
    d_std = cifar_train_dataset.data.std(axis=(0,1,2)) / 255
    normalizer = transforms.Normalize(mean=d_mean, std=d_std)

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        normalizer
    ])

    # Split into validation and test sets
    train_size = int(0.8 * len(cifar_train_dataset))  # 80% for training
    val_size = len(cifar_train_dataset) - train_size   # 20% for validation
    cifar_train_dataset, cifar_val_dataset = torch.utils.data.random_split(cifar_train_dataset, [train_size, val_size])

    # Save datasets for later
    cifar_data_sets = {"train_data": cifar_train_dataset, "val_data": cifar_val_dataset, "test_data": cifar_test_dataset}

    # Create data loaders
    cifar_data_loaders = {}
    cifar_data_loaders["train_data"] = torch.utils.data.DataLoader(cifar_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    cifar_data_loaders["val_data"] = torch.utils.data.DataLoader(cifar_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    cifar_data_loaders["val_train_data"] = torch.utils.data.DataLoader(cifar_train_dataset, batch_size=len(cifar_train_dataset), shuffle=False, num_workers=num_workers)
    cifar_data_loaders["test_data"] = torch.utils.data.DataLoader(cifar_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return cifar_data_loaders, cifar_data_sets, transform

def calculate_maximum_mutual_information_by_labels(dataset: Dataset) -> float:

    """
    Looking at labels to find the maximum possible mutual information between the labels and the data.
    """

    labels = torch.Tensor([label for _, label in dataset])

    label_counts = torch.unique(labels, return_counts=True)[1]
    probabilities = label_counts / labels.shape[0]
    entropy = -torch.sum(probabilities * torch.log2(probabilities)).item()

    return entropy


def calculate_table_memory_capacity(dataset: Dataset, capacity_per_entry: int = 8) -> float:
    data = torch.stack([data for data, _ in dataset])
    return data.shape[0]*data.shape[1]*data.shape[2]*capacity_per_entry

"""
print("Entropy per row:", entropy, "Total entropy:", entropy*mnist_labels.shape[0])
print("Memory capacity of a corresponding table:", eq_memory_cap, "bits")
print("Suggested maximum compression rate:", eq_memory_cap/(entropy*mnist_labels.shape[0]))
"""

    