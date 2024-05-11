from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torch import torch

from utils import prepare_data_loaders
from model import ChineseCNN
import chinese_mnist_loader

def visualize_model(visualization_path, model, sample):
    writer = SummaryWriter(visualization_path)

    writer.add_graph(model, sample)

data, labels = chinese_mnist_loader.load_chinese_mnist() #sample input

transform = prepare_data_loaders(data, labels, train_perc=0.333, test_perc=0.333, batch_size=50, num_workers=4)[2]

model = ChineseCNN()

visualize_model('./log', model, data[1])
