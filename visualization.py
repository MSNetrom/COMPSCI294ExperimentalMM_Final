from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torch import torch.load

from utils import prepare_data_loaders
from model import ChineseCNN
import chinese_mnist_loader

def visualize_model(visualization_path, model_path, sample):
    writer = SummaryWriter(visualization_path)
    model = ChineseCNN()
    model = model.load_state_dict(torch.load(model_path))

    writer.add_graph(model, sample)

    #tensorboard --logdir = %visualization_path% to access visualization locally

data, labels = chinese_mnist_loader.load_chinese_mnist() #sample input

transform = prepare_data_loaders(data, labels, train_perc=0.333, test_perc=0.333, batch_size=50, num_workers=4)[2]

model = ChineseCNN()

visualize_model('/log','/saved_model.pt', data[1])