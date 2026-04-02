# from: https://github.com/IMvision12/SegFormer-tf/blob/main/models/Head.py
# converted to pytorch
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils_3d import ResizeLayer3D


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "lrelu": nn.LeakyReLU,
    "prelu": nn.PReLU,
}


class MLP3D(nn.Module):
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
        super(MLP3D, self).__init__()
        self.proj = nn.Linear(input_dim, decode_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConvModule3D(nn.Module):
    """
    3D Convolutional Module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        act_name (str, optional): Name of activation function. Defaults to 'relu'.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, C, D, H, W), where B is the batch size, 
                          C is the number of input channels, D is depth, H is height, and W is width.

    Outputs:
        x (torch.Tensor): Output tensor of shape (B, out_channels, D, H, W), after applying convolution, 
                      batch normalization, and activation.
    """
    def __init__(self, in_channels: int, out_channels: int, act_name: str="relu"):
        super(ConvModule3D, self).__init__()
        self.act_name = act_name
        act_cls = _ACTIVATIONS.get(act_name.lower())
        if act_cls is None:
            raise ValueError(f"Unknown activation '{act_name}'. Valid keys: {list(_ACTIVATIONS)}")

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.9)
        self.act_cls = act_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x) if self.training else x  # use self.training in training mode
        x = self.act_cls(x)
        return x


class SegFormerHead3D(nn.Module):
    """
    3D SegFormer Head for Segmentation.

    Args:
        input_dims (list of int): List of channel dimensions for each MLP.
        decode_dim (int, optional): The decoding dimension. Defaults to 768.
        num_classes (int, optional): The number of output classes. Defaults to 19.
        act_name (str, optional): Name of activation function, Default to 'relu'.

    Inputs:
        inputs (list of torch.Tensor): A list of 5-dimensional input tensors of shape (B, C, D, H, W),
                                       where B is the batch size, C is the number of input channels,
                                       D is the depth, H is the height, and W is the width.

    Outputs:
        x (torch.Tensor): A 5-dimensional output tensor of shape (B, num_classes, D, H, W) after applying
                      the segmentation head.
    """    
    def __init__(self, input_dims: list, decode_dim: int=768, num_classes: int=19, act_name: str="relu"):
        super(SegFormerHead3D, self).__init__()
        assert input_dims is not None, "input_dims must be a list with the channel dimensions!"

        self.decode_dim = decode_dim
        self.linear_layers = nn.ModuleList([MLP3D(in_dim, decode_dim) for in_dim in input_dims])

        self.linear_fuse = ConvModule3D(in_channels=len(input_dims) * decode_dim, out_channels=decode_dim, act_name=act_name)
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv3d(decode_dim, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        D, H, W = inputs[0].shape[2], inputs[0].shape[3], inputs[0].shape[4]
        outputs = []

        for x, mlp in zip(inputs, self.linear_layers):
            x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
            x = mlp(x)
            x = x.permute(0, 4, 1, 2, 3)  # back to (B, C, D, H, W)
            x = ResizeLayer3D(D, H, W)(x)
            outputs.append(x)

        x = torch.cat(outputs[::-1], dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x) if self.training else x
        x = self.linear_pred(x)

        return x
