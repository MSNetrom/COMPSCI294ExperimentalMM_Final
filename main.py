import torch
import torch.utils

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import (prepare_data_loaders, data_in_table,
                     get_information_per_column, get_mutual_information, unpack_dataset,
                     memory_equivalent_capacity_of_table)
from model import OurCNN

def train_model(cifar_data_loaders, transform, num_epochs=1, name="my_model"):
    """
    Code for actually training a model
    """
    # Create model
    my_model = OurCNN(preparer=transform)

    # Logger and checkpoints saving
    logger = TensorBoardLogger("tb_logs", name=name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_val_set/dataloader_idx_0",
        dirpath=f"checkpoints/{name}",
        filename="my_model-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,  # Save the top 3 models
        mode="min",    # Save models with the minimum train_loss
    )
    
    # Create a PyTorch Lightning Trainer and train the model
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=500,
    )

    # Train the model
    trainer.fit(my_model, cifar_data_loaders["train_data"], [cifar_data_loaders["val_data"], cifar_data_loaders["val_train_data"]])

    # Evaluate the model on the test set
    trainer.test(my_model, cifar_data_loaders["test_data"])

def evaluate_data(data, labels):
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
    print(f"Minimum mutual information between labels and data: {mutual_info.min():.4f} bits")
    

if __name__ == "__main__":

    cifar_data_loaders, cifar_data_sets, transform = prepare_data_loaders()

    data, labels = unpack_dataset(cifar_data_sets["train_data"])
    # Do some evaluation on the data
    evaluate_data(data, labels)

    # Train a model
    #train_model(cifar_data_loaders=cifar_data_loaders, transform=transform, num_epochs=10, name="my_model")