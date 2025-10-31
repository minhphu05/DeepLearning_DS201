import numpy as np
import torch
from torch import nn
from mnist_dataset import collate_fn, MnistDataset
from torch.utils.data import DataLoader
from perceptron_1_layer import Perceptron_1_layer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"

train_dataset = MnistDataset(
    image_path="/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_01/data/raw_data/train-images-idx3-ubyte/train-images-idx3-ubyte",
    label_path="/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_01/data/raw_data/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)

model = Perceptron_1_layer(
    image_size=(28,28),
    num_labels=10
).to(device)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)


for item in train_dataloader:
    image:torch.Tensor = item["image"]
    # image = image.to(device)
    output = model(image)
    print(output)
    print(output.shape)
    raise

