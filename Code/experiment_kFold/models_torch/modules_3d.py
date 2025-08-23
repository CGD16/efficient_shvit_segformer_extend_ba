# from: https://github.com/IMvision12/SegFormer-tf/blob/main/models/modules.py
# converted to pytorch and 3D

import torch
import torch.nn as nn

from models_torch.attention_3d import Attention3D
from models_torch.utils_3d import DropPath3D


class DWConv3D(nn.Module):
    """
    3D Depth-Wise Convolution Layer.

    Args:
        hidden_features (int, optional): The number of hidden features/channels. Defaults to 768.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size, 
                          N is the sequence length, and C is the number of features.
        D (int): Depth dimension of the input tensor.
        H (int): Height dimension of the input tensor.
        W (int): Width dimension of the input tensor.

    Outputs:
        x (torch.Tensor): Output tensor of shape (B, D*H*W, C) after applying the depth-wise convolution.
    """
    def __init__(self, hidden_features: int=768):
        super(DWConv3D, self).__init__()
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        x = x.view(x.size(0), D, H, W, x.size(-1)).permute(0, 4, 1, 2, 3)  # B, C, D, H, W
        x = self.dwconv(x)
        x = x.flatten(2).permute(0, 2, 1)  # B, D*H*W, C
        return x


class Mlp3D(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with Depth-Wise Convolution.

    Args:
        in_features (int): The number of input features.
        hidden_features (int, optional): The number of hidden features. Defaults to in_features.
        out_features (int, optional): The number of output features. Defaults to in_features.
        drop (float, optional): Dropout rate. Defaults to 0.0.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size, 
                          N is the sequence length, and C is the number of input features.
        D (int): Depth dimension of the input tensor.
        H (int): Height dimension of the input tensor.
        W (int): Width dimension of the input tensor.

    Outputs:
        x (torch.Tensor): Output tensor of shape (B, N, out_features) after linear projection, 
                      depth-wise convolution, activation, and dropout.
    """
    def __init__(self, in_features: int, hidden_features: int=None, out_features: int=None, drop: float=0.0):
        super(Mlp3D, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv3D(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, D, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block3D(nn.Module):
    """
    3D Transformer Block with multi-head self-attention and MLP.

    Args:
        dim (int): The number of input dimensions.
        num_heads (int): The number of attention heads.
        mlp_ratio (float, optional): Ratio of hidden dimension to input dimension for the MLP. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to include bias in query, key, and value projections. Defaults to False.
        drop (float, optional): Dropout rate. Defaults to 0.0.
        attn_drop (float, optional): Dropout rate on attention weights. Defaults to 0.0.
        drop_path (float, optional): Dropout rate on the residual path. Defaults to 0.0.
        sr_ratio (int, optional): Spatial reduction ratio for the attention layer. Defaults to 1.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size, 
                          N is the sequence length, and C is the number of input dimensions.
        D (int): Depth dimension of the input tensor.
        H (int): Height dimension of the input tensor.
        W (int): Width dimension of the input tensor.

    Outputs:
        x (torch.Tensor): Output tensor of shape (B, N, C) after attention and MLP operations.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float=4.0, qkv_bias: bool=False, 
                 drop: float=0.0, attn_drop: float=0.0, drop_path: float=0.0, sr_ratio: int=1):
        super(Block3D, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention3D(dim, num_heads, sr_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath3D(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = Mlp3D(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), D, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), D, H, W))
        return x


class OverlapPatchEmbed3D(nn.Module):
    """
    3D Patch Embedding with overlapping patches.

    Args:
        img_size (int, optional): Size of the input image. Defaults to 224.
        img_channels (int, optional): Number of channels in the input image. Defaults to 3.
        patch_size (int, optional): Size of the patch. Defaults to 7.
        stride (int, optional): Stride of the convolution. Defaults to 4.
        filters (int, optional): Number of output filters. Defaults to 768.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, img_channels, H, W), where B is the batch size, 
                          img_channels is the number of input channels,
                          H is the height, and W is the width.

    Outputs:
        x (torch.Tensor): Output tensor of shape (B, N, filters), where N is the reshaped sequence length.
        D (int): Depth dimension of the output tensor.
        H (int): Height dimension of the output tensor.
        W (int): Width dimension of the output tensor.
    """
    def __init__(self, img_size: int=224, img_channels: int=3, patch_size: int=7, stride: int=4, filters: int=768):
        super(OverlapPatchEmbed3D, self).__init__()
        self.conv = nn.Conv3d(img_channels, filters, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(filters, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # B, D*H*W, C
        x = self.norm(x)
        return x, D, H, W


class MixVisionTransformer3D(nn.Module):
    """
    3D Vision Transformer with mixed patch embeddings and multiple Transformer blocks.

    Args:
        img_size (int, optional): Size of the input image. Defaults to 224.
        img_channels (int, optional): Number of input channels. Defaults to 3.
        embed_dims (list of int, optional): Embedding dimensions for each stage. Defaults to [64, 128, 256, 512].
        num_heads (list of int, optional): Number of attention heads for each stage. Defaults to [1, 2, 4, 8].
        mlp_ratios (list of int, optional): MLP ratios for each stage. Defaults to [4, 4, 4, 4].
        qkv_bias (bool, optional): Whether to use bias in query, key, and value projections. Defaults to False.
        drop_rate (float, optional): Dropout rate. Defaults to 0.0.
        attn_drop_rate (float, optional): Dropout rate for attention layers. Defaults to 0.0.
        drop_path_rate (float, optional): Dropout rate for stochastic depth. Defaults to 0.0.
        depths (list of int, optional): Depth (number of blocks) for each stage. Defaults to [3, 4, 6, 3].
        sr_ratios (list of int, optional): Spatial reduction ratios for each stage. Defaults to [8, 4, 2, 1].

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, img_channels, D, H, W), where B is the batch size, 
                          img_channels is the number of input channels,
                          D is the depth, H is the height, and W is the width.

    Outputs:
        list of torch.Tensor: List of output tensors from each stage, where each tensor has shape (B, C, D, H, W).
    """
    def __init__(self, img_size: int=224, img_channels: int=3, embed_dims: list=[64, 128, 256, 512], 
                 num_heads: list=[1, 2, 5, 8], mlp_ratios: list=[4, 4, 4, 4], qkv_bias: bool=False, 
                 drop_rate: float=0.0, attn_drop_rate: float=0.0, drop_path_rate: float=0.0,
                 depths: list=[3, 4, 6, 3], sr_ratios: list=[8, 4, 2, 1]):
        super(MixVisionTransformer3D, self).__init__()
        self.depths = depths
        
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        img_sizes = [img_size, img_size // 4, img_size // 8, img_size // 16]
        channels = [img_channels] + embed_dims[:-1]

        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed3D(img_size=img_sizes[i], img_channels=channels[i], patch_size=patch_sizes[i],
                              stride=strides[i], filters=embed_dims[i])
            for i in range(len(embed_dims))
        ])

        dpr = torch.linspace(0.0, drop_path_rate, sum(depths)).tolist()

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                Block3D(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i]) + j],
                      sr_ratio=sr_ratios[i])
                for j in range(depths[i])
            ])
            for i in range(len(embed_dims))
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims[i], eps=1e-5) for i in range(len(embed_dims))])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        for i, (patch_embed, block, norm) in enumerate(zip(self.patch_embeds, self.blocks, self.norms)):
            x, D, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, D, H, W)
            x = norm(x).permute(0, 2, 1).view(B, -1, D, H, W)
            outs.append(x)

        return outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)