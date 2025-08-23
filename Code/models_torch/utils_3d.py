# from: https://github.com/IMvision12/SegFormer-tf/blob/main/models/utils.py
# converted to pytorch and 3D
 
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResizeLayer3D(nn.Module):
    """
    The 3D ResizeLayer class is to resize 3D input tensors to a fixed set of dimensions 
    (depth, height, and width) using trilinear interpolation.

    Args:
        target_depth (int): The desired depth of the output tensor.
        target_height (int): The desired height of the output tensor.
        target_width (int): The desired width of the output tensor.

    Inputs:
        inputs (torch.Tensor): A 5-dimensional input tensor of shape (N, C, D, H, W),
                            where N is the batch size, C is the number of channels,
                            D is the input depth, H is the input height, and W is the input width.

    Outputs:
        x (torch.Tensor): A 5-dimensional output tensor of shape (N, C, target_depth, target_height, target_width).
                    The tensor is resized using trilinear interpolation.
    """
    def __init__(self, target_depth: int, target_height: int, target_width: int):
        super(ResizeLayer3D, self).__init__()
        self.target_depth = target_depth
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.target_depth, self.target_height, self.target_width), 
                          mode="trilinear", align_corners=True)
        return x


class DropPath3D(nn.Module):
    """
    The DropPath3D class implements the stochastic depth technique, which randomly drops 
    paths during the training of deep neural networks. This helps to regularize the model 
    and prevents overfitting, thereby improving its generalization capabilities.

        Args:
            drop_prob (float, optional): Probability of DropPath (between 0.0 and 1.0).
                               A higher value increases the chance of dropping paths.
            x (torch.Tensor): Input tensor with arbitrary shape.
        
        Returns:
            x (torch.Tensor): Output tensor after applying DropPath.
    """ 
    def __init__(self, drop_prob: float=0.0):
        super(DropPath3D, self).__init__()
        self.drop_prob = torch.clamp(torch.tensor(drop_prob), min=0.0, max=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1 - self.drop_prob
            random_tensor = keep_prob + torch.rand_like(x)  # Random values for Drop
            random_tensor.floor_()  # In-Place-Floor for binary masking
            x = (x / keep_prob) * random_tensor  # Scale to retain expected value
        return x