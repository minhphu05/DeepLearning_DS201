import torch
from torch import nn
from torch.nn import functional as F

class Perceptron_1_layer(nn.Module):
    def __init__(self, image_size: tuple, num_labels: int):
        super().__init__()
        w, h = image_size
        self.linear = nn.Linear(
            in_features=w*h,
            out_features=num_labels
        )
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor):
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        x = self.linear(x)
        output = self.softmax(x)
        
        return output