# from: https://github.com/IMvision12/SegFormer-tf/blob/main/models/Head.py
# converted to pytorch and 3D

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ResizeLayer


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Layer.

    Args:
        input_dim (int): The number of input dimensions.
        decode_dim (int): The number of output dimensions.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, *, input_dim), where B is the batch size, 
                          and * represents any number of additional dimensions.

    Outputs:
        torch.Tensor: Output tensor of shape (B, *, decode_dim), after linear projection.
    """
    def __init__(self, input_dim: int, decode_dim: int):
        super(MLP, self).__init__()
        self.proj = nn.Linear(input_dim, decode_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConvModule(nn.Module):
    """
    2D Convolutional Module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, 
                          C is the number of input channels, H is height, and W is width.

    Outputs:
        torch.Tensor: Output tensor of shape (B, out_channels, H, W), after applying convolution, 
                      batch normalization, and ReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x) if self.training else x  # use self.training in training mode
        x = self.relu(x)
        return x


class SegFormerHead(nn.Module):
    """
    2D SegFormer Head for Segmentation.

    Args:
        input_dims (list of int): List of channel dimensions for each MLP.
        decode_dim (int, optional): The decoding dimension. Defaults to 768.
        num_classes (int, optional): The number of output classes. Defaults to 19.

    Inputs:
        inputs (list of torch.Tensor): A list of 4-dimensional input tensors of shape (B, C, H, W),
                                       where B is the batch size, C is the number of input channels,
                                       H is the height, and W is the width.

    Outputs:
        torch.Tensor: A 4-dimensional output tensor of shape (B, num_classes, H, W) after applying
                      the segmentation head.
    """    
    def __init__(self, input_dims: list, decode_dim: int=768, num_classes: int=19):
        super(SegFormerHead, self).__init__()
        assert input_dims is not None, "input_dims must be a list with the channel dimensions!"

        self.decode_dim = decode_dim
        self.linear_layers = nn.ModuleList([MLP(in_dim, decode_dim) for in_dim in input_dims])

        self.linear_fuse = ConvModule(in_channels=len(input_dims) * decode_dim, out_channels=decode_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(decode_dim, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        H, W = inputs[0].shape[2], inputs[0].shape[3]
        outputs = []
        
        for x, mlp in zip(inputs, self.linear_layers):
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = mlp(x)
            x = x.permute(0, 3, 1, 2)  # back to (B, C, H, W)
            x = ResizeLayer(H, W)(x)
            outputs.append(x)

        x = torch.cat(outputs[::-1], dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x) if self.training else x
        x = self.linear_pred(x)

        return x
