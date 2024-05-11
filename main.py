import torch
import torch.utils

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import utils.chinese_mnist_loader as chinese_mnist_loader
from utils.cnn_size_calcs import full_cnn_seq_size


from utils.utils import (prepare_data_loaders, data_in_table,
                     get_information_per_column, get_mutual_information, unpack_dataset,
                     memory_equivalent_capacity_of_table, load_mnist)

from strategy1_networks import OptimalCNN, HIDDEN_SIZES
from strategy2_networks import STRAT1_TEST_LIST
from fully_connected_extra import FullyConnectedExtra

torch.manual_seed(0)

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

def evaluate_data(data: torch.Tensor, labels: torch.Tensor, classes=15):
    """
    Code for doing some evaluation on the data
    """

    # Calculate the memory capacity of the dataset
    data_capacity = data_in_table(data)
    print(f"Amount of data in the dataset: {data_capacity} bits")

    # Memory Eqviavlent Capacity of table
    memory_equivalent_capacity = memory_equivalent_capacity_of_table(labels, classes)
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

def evaluate_strategy_1(data_loaders, transform, namde_addon="", num_epochs=500):

    #hidden_sizes = [4, 6, 9, 13, 19, 56, 222, 888, 3552]
    hidden_sizes = reversed(HIDDEN_SIZES)

    for h in hidden_sizes:
        model = OptimalCNN(hidden_size=h, preparer=transform)
        train_model(model=model, data_loaders=data_loaders, num_epochs=num_epochs, name=f"Strat1_{h}{namde_addon}")

        with open(f"tb_logs/Strat1_{h}{namde_addon}/mec_info.txt", "w") as f:
            f.write(f"Hidden size: {h}\nMEC: {11*h}")


def evaluate_strategy_2(data_loaders, transform, namde_addon="", num_epochs=500):

    
    for model_class in STRAT1_TEST_LIST:
        model = model_class(input_size=(1, 64, 64), preparer=transform)
        train_model(model=model, data_loaders=data_loaders, num_epochs=num_epochs, name=f"Strat2_{model_class.__name__}{namde_addon}")

def evaluate_fully_connected_extra(data_loaders, transform, num_epochs=500, namde_addon=""):
    model = FullyConnectedExtra(preparer=transform)
    train_model(model=model, data_loaders=data_loaders, num_epochs=num_epochs, name=f"FullyConnectedExtra{namde_addon}")

if __name__ == "__main__":

    chinese_data, chinese_labels = chinese_mnist_loader.load_chinese_mnist()
    chinese_data_loaders, _, chinese_transform = prepare_data_loaders(chinese_data, chinese_labels, train_perc=2/3, test_perc=1/6, batch_size=50, num_workers=4)

    evaluate_strategy_1(chinese_data_loaders, chinese_transform, num_epochs=500, namde_addon="_Chinese")
    evaluate_strategy_2(chinese_data_loaders, chinese_transform, num_epochs=500, namde_addon="_Chinese")
    evaluate_fully_connected_extra(chinese_data_loaders, chinese_transform, num_epochs=500, namde_addon="_Chinese")