import torch
from torch import nn
import torch.nn.functional as F

class Perceptron_3_layer(nn.Module):
    def __init__(self, image_size: int, num_classes: int):
        super().__init__()
        
        w, h = image_size
        
        input_dim = w * h
        
        self.layer1 = nn.Linear(
            in_features=input_dim,
            out_features=512
        )
        self.layer2 = nn.Linear(
            in_features=512,
            out_features=256
        )
        self.layer3 = nn.Linear(
            in_features=256,
            out_features=num_classes
        )

    def forward (self, x: torch.Tensor):
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        
        x = self.layer1(x)
        x = F.relu(x)
        
        x = self.layer2(x)
        x = F.relu(x)
        
        x = self.layer3(x)
        x = F.softmax(x, dim=1)
        
        return x