import torch
import torch.utils
import matplotlib.pyplot as plt
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import utils.chinese_mnist_loader as chinese_mnist_loader

from utils.utils import (prepare_data_loaders, data_in_table,
                     get_information_per_column, get_mutual_information, unpack_dataset,
                     memory_equivalent_capacity_of_table, load_mnist)
from old_files.model import OurCNN, MECCNN, HighDimMECCNN, ChineseCNN
from old_files.capacity_progress_models import DynamicCapacityCNN, DynamicCapacityCNN2
from old_files.optimal_cnn import OptimalCNN, OptimalCNN2, OptimalCNN3, OptimalCNN4

from strategy2_networks import STRAT1_TEST_LIST, CNNBestGuess, Strat2Base, CNNHigher1
from utils.cnn_size_calcs import conv_out_size, full_cnn_seq_size, cnn_mec_calc

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

def chinese_dynamic_train(data_loaders, preparer):

    model = DynamicCapacityCNN2(num_blocks=3, preparer=preparer, end_block=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),          
                            ))
    
    train_model(model, data_loaders=data_loaders, num_epochs=400, name=f"ExtraDynamicCapacityCNN")

    with open(f"checkpoints/ExtraDynamicCapacityCNN/mec_info.txt", "w") as f:
        f.write(f"MEC: {model.get_mec()}")


def chinese_capacity_tester(data_loaders, preparer):

    # Train for 20 000 steps?

    # 32 channels
    models = [
        DynamicCapacityCNN(num_blocks=3, preparer=preparer, end_block=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),          
                            )), # 144 input to linear layer
        DynamicCapacityCNN(num_blocks=3, preparer=preparer, end_block=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=32, out_channels=27, kernel_size=3, stride=1, padding=2),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),          
                            )), # 108 input to linear layer
        DynamicCapacityCNN(num_blocks=3, preparer=preparer, end_block=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=32, out_channels=18, kernel_size=3, stride=1, padding=2),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),          
                            )), # 72 input to linear layer
        DynamicCapacityCNN(num_blocks=3, preparer=preparer, end_block=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=32, out_channels=36, kernel_size=3, stride=1, padding=1),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),          
                            )), # 36 input to linear layer
        DynamicCapacityCNN(num_blocks=3, preparer=preparer, end_block=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),          
                            )), # 4 input size
        DynamicCapacityCNN(num_blocks=3, preparer=preparer, end_block=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),          
                            )) # 2 input size, 2*2 = 4 > log_2(15)
    ]

    for i, m in enumerate(models):
        print("Training model:", i, "MEC:", m.get_mec())
        train_model(m, data_loaders=data_loaders, num_epochs=400, name=f"DynamicCapacityCNN_{i}")

        with open(f"checkpoints/DynamicCapacityCNN_{i}/mec_info.txt", "w") as f:
            f.write(f"MEC: {m.get_mec()}")

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

def tes_optimals():

    data, labels = chinese_mnist_loader.load_chinese_mnist()
    data_loaders, data_sets, transform = prepare_data_loaders(data, labels, train_perc=2/3, test_perc=1/6, batch_size=50, num_workers=4)

    hidden_sizes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 28, 56, 111, 222, 444, 888, 1776, 3552]
    hidden_sizes = reversed(hidden_sizes)

    for h in hidden_sizes:
        model = OptimalCNN4(hidden_size=h, preparer=transform)
        train_model(model=model, data_loaders=data_loaders, num_epochs=600, name=f"ExpOptimalCNN4_{h}")

        with open(f"tb_logs/ExpOptimalCNN4_{h}/mec_info.txt", "w") as f:
            f.write(f"Hidden size: {h}\nMEC: {11*h}")


    

if __name__ == "__main__":

    #data, labels = chinese_mnist_loader.load_chinese_mnist()
    #data_loaders, data_sets, transform = prepare_data_loaders(data, labels, train_perc=2/3, test_perc=1/6, batch_size=50, num_workers=4)

    #model = CNNHigher1(input_size=(1, 64, 64), preparer=transform)

    #train_model(model=model, data_loaders=data_loaders, num_epochs=500, name=f"Strat1BestGuess")

    #in_data = torch.randn((1, 1, 64, 64))

    #for model_class in STRAT1_TEST_LIST:
    #    model = model_class(input_size=(1, 64, 64))
    #    print("MEC:", model.get_mec(), "Params:", model.get_param_count())
        

    #c_mod = CNNBestGuess(input_size=(1, 64, 64))
    #print("MEC:", c_mod.get_mec(), "Params:", c_mod.get_param_count())

    #print("CNN sizes:", full_cnn_seq_size(c_mod.conv_sequence, (1, 64, 64)))
    #print("CNN MEC:", cnn_mec_calc(c_mod.conv_sequence, (1, 64, 64)))
    #print("Full MEC:", c_mod.get_mec())
    #print(c_mod.get_mec())

    #c_mod(in_data)

    #mutual_information_comparer()
    
    # Load the data
    #data, labels = chinese_mnist_loader.load_chinese_mnist()
    #data_loaders, data_sets, transform = prepare_data_loaders(data, labels, train_perc=2/3, test_perc=1/6, batch_size=50, num_workers=4)

    #evaluate_data(*unpack_dataset(data_sets["train_data"]))
    #tes_optimals()

    #model = OptimalCNN4(hidden_size=9, preparer=transform)

    #chinese_dynamic_train(data_loaders, transform)


    #model = ChineseCNN(preparer=transform)
    #train_model(model=model, data_loaders=data_loaders, num_epochs=10000, name="OptimalChineseCNN4_2")
    #data, labels = unpack_dataset(cifar_data_sets["train_data"])
    # Do some evaluation on the data
    #evaluate_data(data, labels)

    #model = HighDimMECCNN(preparer=transform)
    # Train a model
    #train_model(model=model, cifar_data_loaders=cifar_data_loaders, num_epochs=10000, name="HighDimMECCNN")