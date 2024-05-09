import math
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import pandas as pd
from .chinese_mnist_loader import load_chinese_mnist

from typing import Tuple, Dict

def load_titanic() -> Tuple[torch.Tensor, torch.Tensor]:

    # Load the Titanic dataset
    # train_and_test2.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
    df1 = pd.read_csv('titanic_dataset.csv', delimiter=',')
    df1.dataframeName = 'titanic_dataset.csv'

    data = torch.Tensor(df1.values[:, :-1])
    labels = torch.Tensor(df1.values[:, -1])

    return data, labels

def load_cifar10() -> Tuple[Dataset, Dataset]:

    # Load the CIFAR10 dataset
    cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.PILToTensor())
    cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.PILToTensor())

    return cifar_train_dataset, cifar_test_dataset

def load_mnist() -> Tuple[Dataset, Dataset]:

    # Load the MNIST dataset
    mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    return mnist_train_dataset, mnist_test_dataset

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

def mutual_information_image(data, labels, name=None):

    mutual_infomration = get_mutual_information(data.flatten(start_dim=1), labels).reshape(*data.shape[2:])

    print("Mutual information:", mutual_infomration.shape)
    plt.imshow(mutual_infomration, cmap='hot')
    plt.title("Mutual information between pixels and labels")
    cbar = plt.colorbar()
    cbar.set_label('Average mutual Information in bits')
    if name is not None:
        plt.savefig(name, format="pdf")
    plt.show()

def mutual_information_comparer():

    data, labels = chinese_mnist_loader.load_chinese_mnist()
    data_loaders, data_sets, transform = prepare_data_loaders(data, labels, train_perc=0.333, test_perc=0.333, batch_size=50, num_workers=4)

    data = torch.vstack([data for data, _ in data_sets["train_data"]]).unsqueeze(1)
    labels = torch.Tensor([label for _, label in data_sets["train_data"]])

    mutual_information_image(data, labels, name="chinese_mnist_mutual_information.pdf")

    train_mnist_set, _ = load_mnist()
    data_mnist = torch.vstack([data for data, _ in train_mnist_set]).unsqueeze(1)
    labels_mnist = torch.Tensor([label for _, label in train_mnist_set])

    #print("Data shape: ", data_mnist.shape, "Data dtype: ", data_mnist.dtype)
    

    mutual_information_image(data_mnist, labels_mnist)

    random_data = torch.rand_like(data)*256
    random_labels = torch.randint(0, 15, (data.shape[0],))

    mutual_information_image(random_data, random_labels, name="random_mutual_information.pdf")

if __name__ == "__main__":
    # Load titanic
    pass