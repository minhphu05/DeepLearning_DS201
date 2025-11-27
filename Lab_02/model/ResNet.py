import torch
from torch import nn
from torch.nn import Module

class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            kernel_size=(3,3), 
            out_channels=out_channels,
            padding=(1,1),
            stride=stride,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3,3),
            padding=(1,1),
            stride=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        # There are 2 types of skip connection / residential connection: identity and projection
        self.shortcut = nn.Sequential()
        
        # If size or channel changes, need Conv 1x1
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1,1),
                    padding=(0,0),
                    stride=stride
                ),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out
        
        
class ResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 21 # Number of labels in VinaFood Dataset
    ) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            stride=2,
            padding=(3,3),
            kernel_size=(7,7)
        )
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=2,
            padding=(1,1)
        )
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(
            in_features = 512, # AvgPool --> (1,1,512)
            out_features=num_classes
        )
        
    def _make_layer(
        self,
        block: int,
        out_channels: int,
        num_blocks: int,
        stride: int
    ):
        '''
            Có 4 stages (layers), mỗi stage gồm 2 blocks
                + Layer 1: Output_size = 56 x 56
                + Layer 2: Output_size = 28 x 28
                + Layer 3: Output_size = 14 x 14
                + Layer 3: Output_size = 7 x 7
        '''
        # Initialize list of strides for blocks 
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append (
                block(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride
                )
            )
            self.in_channels = out_channels
        
        # Toán tử giải nén: *[list] --> *[1,2,3] = 1 2 3
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)    # Conv -> batchnorm -> ReLU
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        return out
        
                
        