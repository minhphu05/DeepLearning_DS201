from torch.nn import Module
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import torch

class LeNet (Module):
    def __init__(self, n_classes=10):
        super(LeNet, self).__init__() # super().__init__()
        # Convolutional Layer 1 : 6 filters, kernel 5x5, pad = 2
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Số ma trận chồng lên nhau 
            out_channels=6,     # số lượng feature map chồng lên nhau
            kernel_size=(5,5),
            stride=1,
            padding=(2,2),
            dilation=1,         # Độ giãn giữa các phần tử trong kernel
            bias=True,
            padding_mode='zeros'
        )
            # Output: 6 ma trận feature map chồng nhau, size = (28, 28, 6)
        
        # Pooling Layer 1
        self.pooling1 = nn.AvgPool2d(
            kernel_size=(2,2),
            stride=2,
            padding=(0,0),
            ceil_mode=False,    # Nếu True, làm tròn output_size lên
        )
            # Output: (14, 14, 6)
        
        # Convolutional Layer 2:
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5,5),
            padding=(0,0),
            stride=1,
            bias=True
        ) 
            # Output: (10, 10, 16)
        
        # Pooling Layer 2
        self.pooling2 = nn.AvgPool2d(
            kernel_size=(2,2),
            stride=2, 
            padding=(0,0)
        )
            # Output: (5, 5, 16)
        
        # Fully Connected
            # 120
        self.fc1 = nn.Linear(
            in_features=5*5*16,
            out_features=120,
            bias=True
        )
        
            # 84
        self.fc2 = nn.Linear(
            in_features=120,
            out_features=84,
            bias=True
        )
        
            # 10
        self.fc3 = nn.Linear(
            in_features=84,
            out_features=n_classes,
            bias=True
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, images: torch.Tensor):
        # (batch_size, channels, height, width) = NCHW
        # images (batch_size, 28, 28)
        images = images.unsqueeze (1) # images (batch_size, 1, 28, 28)
       
        features = self.sigmoid(self.conv1(images))     # (bs, 6, 28, 28)
        features = self.pooling1(features)              # (bs, 6, 14, 14)
        
        features = self.sigmoid(self.conv2(features))   # (bs, 16, 10, 10)
        features = self.pooling2(features)              # (bs, 16, 5, 5) 
        
        features = features.view(features.size(0), -1)  # flatten -> (bs, 5*5*16)
        # features = torch.flatten(features, 1)
        
        features = self.sigmoid(self.fc1(features))
        features = self.sigmoid(self.fc2(features))
        
        output = self.fc3(features)
        
        return output