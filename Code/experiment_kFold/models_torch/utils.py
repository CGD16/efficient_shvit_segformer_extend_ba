# from: https://github.com/IMvision12/SegFormer-tf/blob/main/models/utils.py
# converted to pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResizeLayer(nn.Module):
    """
    The ResizeLayer class is to resize 2D input tensors to a fixed set of dimensions 
    (height, and width) using bilinear interpolation.

    Args:
        target_height (int): The desired height of the output tensor.
        target_width (int): The desired width of the output tensor.

    Inputs:
        inputs (torch.Tensor): A 4-dimensional input tensor of shape (N, C, H, W),
                            where N is the batch size, C is the number of channels,
                            H is the input height, and W is the input width.

    Outputs:
        torch.Tensor: A 4-dimensional output tensor of shape (N, C, target_height, target_width).
                    The tensor is resized using trilinear interpolation.
    """
    def __init__(self, target_height: int, target_width: int):
        super(ResizeLayer, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.target_height, self.target_width), mode="bilinear", align_corners=False)
        return x


class DropPath(nn.Module):
    """
    The DropPath3D class implements the stochastic depth technique, which randomly drops 
    paths during the training of deep neural networks. This helps to regularize the model 
    and prevents overfitting, thereby improving its generalization capabilities.

        Args:
            drop_prob (float, optional): Probability of DropPath (between 0.0 and 1.0).
                               A higher value increases the chance of dropping paths.
            x (torch.Tensor): Input tensor with arbitrary shape.
        
        Returns:
            torch.Tensor: Output tensor after applying DropPath.
    """ 
    def __init__(self, drop_prob: float=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = torch.clamp(torch.tensor(drop_prob), min=0.0, max=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1 - self.drop_prob
            random_tensor = keep_prob + torch.rand_like(x)  # Random values for Drop
            random_tensor.floor_()  # In-Place-Floor for binary masking
            x = (x / keep_prob) * random_tensor  # Scale to retain expected value
        return x