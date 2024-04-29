import torch
from utils import prepare_data_loaders, calculate_maximum_mutual_information_by_labels
from tqdm import tqdm

cifar_data_loaders, cifar_data_sets, transform = prepare_data_loaders()

"""
Chapter 4
"""
def get_information_per_column(data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

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

data = torch.stack([data for data, _ in cifar_data_sets["train_data"]]).flatten(start_dim=1)
labels = torch.Tensor([label for _, label in cifar_data_sets["train_data"]])

#print("Data shape: ", data.shape, "Labels shape: ", labels.shape)

# Calculate information for each feature

label_information = get_information_per_column(labels)
mutual_info = get_mutual_information(data, labels)

print("Label information: ", label_information)
print("Minimum Mutual Information: ", mutual_info.min())

