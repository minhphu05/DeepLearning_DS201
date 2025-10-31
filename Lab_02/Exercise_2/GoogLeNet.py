import torch
from torch import nn
import torch.nn.functional as F


class BasicConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: any
    )->None:
        
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False, 
            **kwargs
        )
        self.ReLU = nn.ReLU(True)
        self.batchnorm2d = nn.BatchNorm2d(
            out_channels,
            eps=0.001
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.batchnorm2d(y)
        y = self.ReLU(y)
        return y

        
class Inception(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        # output channels 
        ch1x1: int,             
        ch3x3_reduced: int,     
        ch3x3: int,
        ch5x5_reduced: int,
        ch5x5: int,
        pool_proj: int
    ) -> None:
        super(Inception, self).__init__()
        
        # First Branch : 1x1 Conv
        self.branch1 = BasicConv2D(
            in_channels=in_channels,
            out_channels=ch1x1,
            kernel_size=(1,1),
            padding=(0,0),
            stride=1
        )
        # Second Branch: 1x1 Conv and 3x3 Conv
        self.branch2 = nn.Sequential(
            BasicConv2D(
                in_channels=in_channels,
                out_channels=ch3x3_reduced,
                kernel_size=(1,1),
                padding=(0,0),
                stride=1
            ),
            BasicConv2D(
                in_channels=ch3x3_reduced,
                out_channels=ch3x3,
                kernel_size=(3,3),
                padding=(1,1),
                stride=1
            )
        )
        # Third Branch: 1x1 Conv and 5x5 Conv
        self.branch3 = nn.Sequential(
            BasicConv2D(
                in_channels=in_channels,
                out_channels=ch5x5_reduced,
                kernel_size=(1,1),
                padding=(0,0),
                stride=1
            ),
            BasicConv2D(
                in_channels=ch5x5_reduced,
                out_channels=ch5x5,
                padding=(2,2),
                stride=1,
                kernel_size=(5,5)
            )
        )
        # Fourth Branch: 3x3 MaxPool and 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=(3,3),
                padding=(1,1),
                stride=1,
                ceil_mode=True
            ),
            BasicConv2D(
                in_channels=in_channels,
                out_channels=pool_proj,
                kernel_size=(1,1),
                padding=(0,0),
                stride=1
            )
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        output = [branch1, branch2, branch3, branch4]
        output = torch.cat(output, 1)   # concatenation
        return output
    
class InceptionAux(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        n_classes: int,
        dropout: float = 0.7
    ):
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AvgPool2d(
            kernel_size=(5,5),
            stride=3,
            padding=(0,0)
        )
        self.conv = BasicConv2D(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=(1,1),
            padding=(0,0),
            stride=1
        )
        self.ReLU = nn.ReLU(True)
        self.fc1 = nn.Linear(
            in_features=2048,
            out_features=1024
        )
        self.fc2 = nn.Linear(
            in_features=1024,
            out_features=n_classes
        )
        self.dropout = nn.Dropout2d(
            p=dropout,
            inplace=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.avgpool(x)
        out = self.conv(out)
        out = torch.flatten(out, 1)
        out = self.ReLU(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
class GoogLeNet(nn.Module):
    def __init__(
        self,
        aux_logits: bool = False,
        n_classes: int = 1000,
        dropout: float = 0.4,
        dropout_aux: float = 0.7
    ) -> None:
        super(GoogLeNet, self).__init__()
        self.aux_logit = aux_logits      # Tag Auxiliary Classifier (True/False)
        self.dropout_aux = dropout_aux

        self.conv1 = BasicConv2D(in_channels=3, out_channels=64, kernel_size=(7,7), stride=2, padding=(3,3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=(0,0))
        self.conv2 = BasicConv2D(in_channels=64, out_channels=64, kernel_size=(1,1), stride=1, padding=(0,0))
        self.conv3 = BasicConv2D(in_channels=64, out_channels=192, kernel_size=(3,3), stride=1, padding=(0,0))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=(0,0), ceil_mode=True)
        
        self.inception3a = Inception(in_channels=192, ch1x1=64, ch3x3_reduced=96, ch3x3=128, ch5x5_reduced=16, ch5x5=32, pool_proj=32) # Output: 28x28x256
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64) # Output : 28x28x480
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=(0,0), ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3,3), padding=(0,0), stride=2, ceil_mode=True)
        
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        
        if aux_logits:
            self.aux1 = InceptionAux(512, n_classes, dropout_aux)
            self.aux2 = InceptionAux(528, n_classes, dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None
        
        # self.avgpool = nn.AvgPool2d(kernel_size=(7,7), stride=1, padding=(0,0))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(in_features=1024, out_features=n_classes)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool2(out)
        
        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool3(out)
        
        out = self.inception4a(out)
        aux1 = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(out)
                
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        aux2 = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(out)
        out = self.inception4e(out)
        out = self.maxpool4(out)
        
        out = self.inception5a(out)
        out = self.inception5b(out)
        out = self.avgpool(out)
        out = self.dropout(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out
        
        
        
