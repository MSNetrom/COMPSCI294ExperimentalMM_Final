import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def load_chinese_mnist(base_path: Path = Path("chinese_mnist")):
    csv = pd.read_csv(base_path / 'chinese_mnist.csv')
    filename = csv[['suite_id', 'sample_id', 'code']].values

    images = [ Image.open(base_path / f"data/data/input_{suite_id}_{sample_id}_{code}.jpg") for suite_id, sample_id, code in filename ]
    labels = [ [x - 1] for x in csv['code'].values ] # need to compensate to 0-15

    image_transform = transforms.PILToTensor()
    images = torch.vstack([ image_transform(image) for image in images ]).unsqueeze(1)
    labels = torch.tensor(labels).flatten()

    print("Images shape: ", images.shape, "Images dtype: ", images.dtype, "Labels shape: ", labels.shape, "Labels dtype: ", labels.dtype)

    return images, labels
#print("Images shape: ", images.shape, "Images dtype: ", images.dtype)
#print("Labels shape: ", labels.shape, "Labels dtype: ", labels.dtype)
