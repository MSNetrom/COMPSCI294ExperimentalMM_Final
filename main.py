import torch
import torch.utils

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import chinese_mnist_loader

from utils import (prepare_data_loaders, data_in_table,
                     get_information_per_column, get_mutual_information, unpack_dataset,
                     memory_equivalent_capacity_of_table)
from model import OurCNN, MECCNN, HighDimMECCNN, ChineseCNN

def train_model(model: torch.nn.Module, data_loaders, num_epochs=1, name="my_model"):
    """
    Code for actually training a model
    """

    # Logger and checkpoints saving
    logger = TensorBoardLogger("tb_logs", name=name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_val_set/dataloader_idx_0",
        dirpath=f"checkpoints/{name}",
        filename=name + "-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,  # Save the top 3 models
        mode="min",    # Save models with the minimum train_loss
    )
    
    # Create a PyTorch Lightning Trainer and train the model
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(model, data_loaders["train_data"], [data_loaders["val_data"], data_loaders["val_train_data"]])

    # Evaluate the model on the test set
    trainer.test(model, data_loaders["test_data"])

def evaluate_data(data: torch.Tensor, labels: torch.Tensor):
    """
    Code for doing some evaluation on the data
    """

    # Calculate the memory capacity of the dataset
    data_capacity = data_in_table(data)
    print(f"Amount of data in the dataset: {data_capacity} bits")

    # Memory Eqviavlent Capacity of table
    memory_equivalent_capacity = memory_equivalent_capacity_of_table(labels, 10)
    print(f"Memory Equivalent Capacity of table as lookup-table: {memory_equivalent_capacity} bits")

    # Calculate the maximum mutual information between the labels and the data
    label_info = get_information_per_column(labels).item() * data.shape[0]
    print(f"Maximum mutual information between labels and data: {label_info:.4f} bits")

    # Calculate the mutual information between the labels and the data
    mutual_info = get_mutual_information(data, labels) * data.shape[0]
    mutual_info_max = mutual_info.max()
    print(f"Minimum mutual information between labels and data: {mutual_info_max:.4f} bits")
    print(f"Meaning we need to memorize {label_info - mutual_info_max:.4f} bits of information to predict the labels")

    # Get mutual information of best guess towards others
    return {"memory_equivalent_capacity": memory_equivalent_capacity}
    

if __name__ == "__main__":

    # Load the data
    data, labels = chinese_mnist_loader.load_chinese_mnist()

    data_loaders, data_sets, transform = prepare_data_loaders(data, labels, train_perc=0.333, test_perc=0.333, batch_size=50, num_workers=4)


    model = ChineseCNN(preparer=transform)
    train_model(model=model, data_loaders=data_loaders, num_epochs=10000, name="ChineseCNN")
    #data, labels = unpack_dataset(cifar_data_sets["train_data"])
    # Do some evaluation on the data
    #evaluate_data(data, labels)

    #model = HighDimMECCNN(preparer=transform)
    # Train a model
    #train_model(model=model, cifar_data_loaders=cifar_data_loaders, num_epochs=10000, name="HighDimMECCNN")