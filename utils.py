import math
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np

from typing import Tuple, Dict

def load_cifar10() -> Tuple[Dataset, Dataset]:

    # Load the CIFAR10 dataset
    cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.PILToTensor())
    cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.PILToTensor())

    return cifar_train_dataset, cifar_test_dataset

def prepare_data_loaders(data: torch.Tensor, labels: torch.Tensor, train_perc: int, test_perc: int, batch_size=50, num_workers=4) -> Tuple[Dict[str, DataLoader], Dict[str, Dataset], transforms.Normalize]:

    #print("Data shape: ", data.shape, "Data dtype: ", data.dtype)
    data = data.float()

    # Create a dataset
    data_set = torch.utils.data.TensorDataset(data, labels)

    # Split into validation and test sets
    train_size = int(train_perc * len(data))
    test_size = int(test_perc * len(data))
    val_size = len(data) - train_size - test_size

    # Split the data
    train_dataset, val_dataset, test_dataset = random_split(data_set, [train_size, val_size, test_size])

    # Create a normalizer for better training
    d_unpack = torch.vstack([my_data for my_data, _ in train_dataset])
    d_unpack = d_unpack.reshape(d_unpack.shape[0], -1, d_unpack.shape[-2], d_unpack.shape[-1])
    print("Data shape: ", d_unpack.shape)
    d_mean = d_unpack.mean(axis=(0,2,3))
    d_std = d_unpack.std(axis=(0,2,3))
    normalizer = transforms.Normalize(mean=d_mean, std=d_std)

    print("Mean: ", d_mean, "Std: ", d_std)

    transform = transforms.Compose([
        #transforms.ConvertImageDtype(torch.float),
        normalizer
    ])

    # Save datasets for later
    data_sets = {"train_data": train_dataset, "val_data": val_dataset, "test_data": test_dataset}

    # Create data loaders
    data_loaders = {}
    data_loaders["train_data"] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    data_loaders["val_data"] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    data_loaders["val_train_data"] = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=num_workers)
    data_loaders["test_data"] = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return data_loaders, data_sets, transform


def data_in_table(data: torch.Tensor, capacity_per_entry: int = 8) -> float:
    return data.shape[0]*data.shape[1]*capacity_per_entry

def memory_equivalent_capacity_of_table(labels: torch.Tensor, classes: int) -> float:
    return labels.shape[0] * math.log2(classes)

def unpack_dataset(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.stack([data for data, _ in dataset]).flatten(start_dim=1)
    labels = torch.Tensor([label for _, label in dataset])
    return data, labels

def get_information_per_column(data: torch.Tensor) -> torch.Tensor:

    if len(data.shape) == 1:
        data = data.unsqueeze(1)

    # Calculate the entropy for each column
    entropies = []
    for i in range(data.size(1)):  # Iterate over columns instead of rows
        column = data[:, i]  # Extract the ith column
        _, counts = torch.unique(column, return_counts=True)
        probabilities = counts / column.shape[0]
        entropy = -torch.sum(probabilities * torch.log2(probabilities))
        entropies.append(entropy.item())
    
    # Convert to a tensor for the entire dataset
    entropies_tensor = torch.tensor(entropies)
    
    return entropies_tensor

def get_mutual_information(data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

    # Calculate the entropy of the datarows, and labels
    information_labels = get_information_per_column(labels)
    information_per_feature = get_information_per_column(data)

    # Calculate the joint entropy
    information_joint = torch.zeros(data.size(1))
    for i in tqdm(range(data.size(1)), desc="Calculating joint entropies"):
        column = data[:, i]
        joint_column = torch.hstack((column.unsqueeze(1), labels.unsqueeze(1))).unsqueeze(1)
        information_joint[i] = get_information_per_column(joint_column)[0]

    # Calculate the mutual information
    return information_per_feature + information_labels - information_joint

    