##########################################################################
#
# small modification of shvit.py
# (with help of pytorch docu, chatgpt, and try&error ...)
# original: https://github.com/ysj9909/SHViT/blob/main/model/shvit.py
#
##########################################################################
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from typing import List

from .center_attention import CenterAttention
 
   
class GroupNorm(nn.GroupNorm):
    """
    The GroupNorm class implements Group Normalization for a given input tensor with shape [B, C, H, W].
    The GroupNorm normalizes the input tensor over the spatial dimensions (height and width) while
    keeping the channel dimension intact. It computes the mean and variance across the spatial dimensions.
 
    Args:
        num_channels (int): Number of channels in the input tensor.
 
    Inputs:
        inputs (Tensor): Input tensor of shape [B, C, H, W] where B is batch size, C is the number of channels,
                         H is height, and W is width.
 
    Outputs:
        Tensor: Normalized tensor of the same shape as the input [B, C, H, W].
    """
    def __init__(self, num_channels: int, **kwargs):
        super(GroupNorm, self).__init__(num_groups=1, num_channels=num_channels, eps=1e-5, **kwargs)


class Conv2d_BN(nn.Sequential):
    """
    The Conv2d_BN class implements a 2D Convolutional layer followed by Batch Normalization
 
    Args:
        in_channels (int): Number of input channels.
        out_channels (int, optional): Number of output channels. Default is 16.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 1.
        strides (int, optional): Stride size of the convolution. Default is 1.
        padding (str, optional): Padding method, either 0 or 1. Default is 0.
        dilation_rate (int, optional): Dilation rate for dilated convolution. Default is 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
 
    Inputs:
        x (Tensor): Input tensor of shape [B, C, H, W] where B is batch size, C is the number of input channels,
                    H is height, and W is width.
 
    Outputs:
        Tensor: Output tensor after applying convolution, batch normalization, and activation function of the same shape
                except the number of channels will be equal to the 'filters' argument [B, filters, H, W].
    """
    def __init__(self, in_channels:int, out_channels: int=16, kernel_size: int=1, strides: int=1, padding: int=0,
                 dilation_rate: int=1, groups: int=1, bn_weight_init: float=1.0):
        super(Conv2d_BN, self).__init__()
        self.add_module("c", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=strides, padding=padding, dilation=dilation_rate, groups=groups, bias=False))
        self.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        nn.init.constant_(tensor=self.bn.weight, val=bn_weight_init)
        nn.init.constant_(tensor=self.bn.bias, val=0)
 
    @torch.no_grad()
    def fuse(self):
        """
        Fuse the convolution and batch normalization layers into a single convolutional layer.
 
        Outputs:
            nn.Conv2d: A new convolutional layer that integrates the batch normalization parameters.
        """        
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:], stride=self.c.stride,
                      padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups, device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


# copy from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/helpers.py
def make_divisible(v: int=None, divisor: int=8, min_value: float=None, round_limit: float=0.9):
    """
    The make_divisible function is used to ensure that a value is divisible by a specified divisor.
    This is often used in neural network architectures where certain quantities (e.g., number of channels)
    need to be divisible by a specific number.
 
    Args:
        v (int or float): The value to be made divisible.
        divisor (int, optional): The divisor to make the value divisible by. Default is 8.
        min_value (float or None, optional): The minimum value to consider. If None, it is set to the value of divisor.
        round_limit (float, optional): A threshold to ensure rounding down does not reduce the value by more than 10%. Default is 0.9.
 
    Returns:
        int: The adjusted value that is divisible by the divisor.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return int(new_v)

 
# original from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/squeeze_excite.py)
class SqueezeExcite(nn.Module):
    """
    The SqueezeExcite class implements the Squeeze-and-Excitation (SE) module, as defined in the original SE-Nets paper, with a few additions.
    It performs channel-wise feature recalibration by aggregating feature maps, learning channel-wise dependencies, and
    then using this information to recalibrate the input feature map.
   
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer    
 
    Args:
        channels (int): Number of input channels.
        rd_ratio (float, optional): Reduction ratio for calculating the number of reduced channels. Default is 1/16.
        rd_channels (int or None, optional): Number of reduced channels. If None, it is calculated using rd_ratio and rd_divisor.
        rd_divisor (int, optional): Divisor for rounding the number of reduced channels. Default is 8.
        add_maxpool (bool, optional): If True, adds global max pooling to the squeeze aggregation. Default is False.
        bias (bool, optional): If True, adds a bias term to the convolutional layers. Default is True.
        norm_layer (callable or None, optional): Normalization layer to use after the first convolution. If None, no normalization is applied. Default is None.
 
    Inputs:
        x (Tensor): Input tensor of shape [B, C, H, W] where B is batch size, C is the number of input channels,
                    H is height, and W is width.
                   
    Outputs:
        Tensor: Output tensor after applying the SE module with the same shape as the input [B, C, H, W].        
    """
    def __init__(self, channels: int, rd_ratio: float=1/16, rd_channels: int=None, rd_divisor: int=8, add_maxpool: bool=False,
            bias: bool=True, norm_layer=None):
        super(SqueezeExcite, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(int(channels * rd_ratio), rd_divisor)
       
        self.fc1 = nn.Conv2d(in_channels=channels, out_channels=rd_channels, kernel_size=1)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels=rd_channels, out_channels=channels, kernel_size=1)
        self.gate = nn.Sigmoid()
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = F.adaptive_avg_pool2d(x, 1)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * F.adaptive_max_pool2d(x, 1)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
   
 
class PatchMerging(nn.Module):
    """
    The PatchMerging class implements a patch merging mechanism using three Conv2d_BN layers and an activation function.
    It processes the input through these layers sequentially, applying ReLU activations and a Squeeze-and-Excitation module
    before the final convolution.
 
    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
 
    Inputs:
        inputs (Tensor): Input tensor of shape [B, C, H, W] where B is batch size, C is the number of input channels (dim),
                         H is height, and W is width.
 
    Outputs:
        Tensor: Output tensor after processing with the same height and width, but with 'out_dim' number of channels [B, out_dim, H/2, W/2].
    """
    def __init__(self, dim: int, out_dim: int):
        super(PatchMerging, self).__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(in_channels=dim, out_channels=hid_dim, kernel_size=1, strides=1, padding=0)
        self.act = nn.ReLU()
        self.conv2 = Conv2d_BN(in_channels=hid_dim, out_channels=hid_dim, kernel_size=3, strides=2, padding=1, groups=hid_dim)
        self.se = SqueezeExcite(channels=hid_dim, rd_ratio=0.25)
        self.conv3 = Conv2d_BN(in_channels=hid_dim, out_channels=out_dim, kernel_size=1, strides=1, padding=0)
 
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv1(inputs)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.se(x)
        x = self.conv3(x)
        return x


class Residual(nn.Module):
    """
    The Residual class implements a residual connection for a given layer with an optional dropout.

    Args:
        m (nn.Module): The main layer to apply the residual connection to.
        drop (float, optional): Dropout probability. Default is 0.0.
 
    Inputs:
        x (Tensor): Input tensor of shape [B, C, H, W] where B is batch size, C is the number of channels,
                    H is height, and W is width.
 
    Outputs:
        Tensor: Output tensor after applying the residual connection and optional dropout, with the same shape as the input [B, C, H, W].
    """
    def __init__(self, m: nn.Module, drop: float=0.0):
        super(Residual, self).__init__()
        self.m = m
        self.drop = drop
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop > 0 and self.training:
            # Apply dropout only during training
            mask = torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
            return x + self.m(x)*mask
        else:
            return x + self.m(x)
   
    @torch.no_grad()
    def fuse(self):
        """
        Fuse the main layer's convolution and batch normalization layers into a single convolutional layer.
 
        Returns:
            nn.Module: A fused convolutional layer if the main layer is Conv2d_BN. Otherwise, returns self.
        """
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self        
 
class FFN(nn.Module):
    """
    The FFN class implements a Feed-Forward Network with two Conv2d_BN layers (point-wise convolutions) and a ReLU activation.
 
    Args:
        ed (int): Number of input channels.
        h (int): Number of hidden channels.
 
    Inputs:
        inputs (Tensor): Input tensor of shape [B, C, H, W] where B is batch size, C is the number of input channels (ed),
                         H is height, and W is width.
 
    Outputs:
        Tensor: Output tensor after applying the feed-forward network with the same shape as the input [B, C, H, W].
    """    
    def __init__(self, ed: int, h: int):
        super(FFN, self).__init__()
        self.pw1 = Conv2d_BN(in_channels=ed, out_channels=h)  # First point-wise convolution with BN
        self.act = nn.ReLU()  # ReLU activation
        self.pw2 = Conv2d_BN(in_channels=h, out_channels=ed, bn_weight_init=0)  # Second point-wise convolution with BN
 
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.pw1(inputs)
        x = self.act(x)
        x = self.pw2(x)
        return x
   
   
class SHSA(nn.Module):
    """
    Single-Head Self-Attention
 
    Initialization:
        Initializes scaling factor, dimensions, normalization layer, and the query-key-value (QKV) convolutional layer,
        along with a projection layer.
 
    Args:
        dim (int): The number of input channels.
        qk_dim (int): The dimension of query and key tensors.
        pdim (int): The partial dimension of the input tensor to be split and processed separately.
       
    Inputs:
        x (torch.Tensor): Input tensor with shape (B, C, H, W) where B is batch size,
                          C is the number of channels, H is height, and W is width.
 
    Outputs:
        torch.Tensor: Output tensor with the same shape as the input.        
    """
    def __init__(self, dim: int, qk_dim: int, pdim: int):
        super(SHSA, self).__init__()
 
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
 
        self.pre_norm = GroupNorm(num_channels=pdim)
        self.qkv = Conv2d_BN(in_channels=pdim, out_channels=qk_dim*2 + pdim)
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(in_channels=dim, out_channels=dim, bn_weight_init=0.0))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
       
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim = 1))
 
        return x
    
    
class SHCSA(nn.Module):
    """
    Single-Head Center-Self-Attention 
 
    Initialization:
        Initializes scaling factor, dimensions, normalization layer, and the query-key-value (QKV) convolutional layer,
        along with a projection layer.
 
    Args:
        dim (int): The number of input channels.
        qk_dim (int): The dimension of query and key tensors.
        pdim (int): The partial dimension of the input tensor to be split and processed separately.
        kernel_size (int, default=3): Size of the convolutional kernel used in attention.
       
    Inputs:
        x (torch.Tensor): Input tensor with shape (B, C, D, H, W) where B is batch size,
                          C is the number of channels, D is depth, H is height, and W is width.
 
    Outputs:
        x (torch.Tensor): Output tensor with the same shape as the input.        
    """
    def __init__(self, dim: int, qk_dim: int, pdim: int, kernel_size: int=3):
        super(SHCSA, self).__init__()
 
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
 
        self.pre_norm = GroupNorm(num_channels=pdim)
        self.qkv = Conv2d_BN(in_channels=pdim, out_channels=qk_dim*2 + pdim)
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(in_channels=dim, out_channels=dim, bn_weight_init=0.0))
        self.center_att = CenterAttention(dim=pdim, kernel_size=kernel_size)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        
        x1 = x1.flatten(2).permute(0,2,1)
        x1 = self.center_att(x1, H, W)        
        x1 = x1.reshape(B, H, W, self.pdim).permute(0,3,1,2)

        x = self.proj(torch.cat([x1, x2], dim = 1))
 
        return x
 

class BasicBlock(nn.Module):
    """
    Basic Block for SHViT
 
    Initialization:
        For "s" (later stages): Initializes convolution, self-attention mixer, and feed-forward network (FFN) wrapped in residuals.
        For "i" (early stages): Initializes convolution and FFN as before but uses an identity layer for the mixer.
   
    Args:
        dim (int): The number of input channels.
        qk_dim (int): The dimension of query and key tensors.
        pdim (int): The partial dimension of the input tensor to be split and processed separately.
        block_type (str): The type of block ('s' for later stages, 'i' for early stages).
        kernel_size (int, default=3): Size of the convolutional kernel used in attention.
   
    Forward Pass:
        Calls the convolution layer, the mixer, and the feed-forward network sequentially, returning the output.
   
    Inputs:
        x (torch.Tensor): Input tensor with shape (B, C, H, W) where B is batch size,
                          C is the number of channels, H is height, and W is width.
 
    Outputs:
        torch.Tensor: Output tensor with the same shape as the input.
    """
    def __init__(self, dim: int, qk_dim: int, pdim: int, block_type: str, flg_is_last_stage: bool, 
                 use_center_att: bool, kernel_size: int=3):
        super(BasicBlock, self).__init__()
        if block_type == "s":  # for later stages
            self.conv = Residual(Conv2d_BN(in_channels=dim, out_channels=dim, kernel_size=3, strides=1, padding=1, groups=dim, bn_weight_init=0.0))
            if use_center_att and not flg_is_last_stage:
                self.mixer = Residual(SHCSA(dim=dim, qk_dim=qk_dim, pdim=pdim, kernel_size=kernel_size))
            else:
                self.mixer = Residual(SHSA(dim=dim, qk_dim=qk_dim, pdim=pdim))
            self.ffn = Residual(FFN(ed=dim, h=int(dim * 2)))
        elif block_type == "i":  # for early stages
            self.conv = Residual(Conv2d_BN(in_channels=dim, out_channels=dim, kernel_size=3, strides=1, padding=1, groups=dim, bn_weight_init=0.0))
            self.mixer = nn.Identity()
            self.ffn = Residual(FFN(ed=dim, h=int(dim * 2)))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.mixer(x)
        x = self.ffn(x)
        return x
   
 
#################################################################
class SHViT(nn.Module):
    """
    The SHViT class implements a vision transformer with hierarchical stages, patch embedding, and basic blocks.
 
    Args:
        in_channels (int, optional): Number of input channels. Default is 3.
        embed_dim (List[int], optional): List of embedding dimensions for each stage. Default is [224, 336, 448].
        partial_dim (List[int], optional): List of partial dimensions (proportional to embed_dim) for each stage. Default is [48, 72, 96] with r=1/4.67.
        depth (List[int], optional): Number of blocks at each stage. Default is [4, 7, 6].
        types (List[str], optional): Block types for each stage, "i" for initial, "s" for subsequent. Default is ["i", "s", "s"].
        qk_dim (List[int], optional): List of query-key dimensions for self-attention at each stage. Default is [16, 16, 16].
        down_ops (List[List], optional): List of downsample operations for each stage. Default is [["subsample", 2], ["subsample", 2], [""]].
        num_convs (int, optional): Number of conv2d_bn layers in head. Default is 2.
        num_stages (int, optional): Number of stages. Default is 3.
        use_center_att (bool, optional): Whether to use center_attention in 2nd stage. Defaults to False.
        kernel_size (int, default=3): Size of the convolutional kernel used in attention.
 
    Inputs:
        inputs (Tensor): Input tensor of shape [B, C, H, W] where B is batch size, C is the number of input channels,
                         H is height, and W is width.
 
    Outputs:
        List[Tensor]: List of output tensors from each stage. Each tensor has the same height and width as the input,
                      but the number of channels will vary according to the stage's configuration.
    """
    def __init__(self, in_channels: int=3, embed_dims=[224, 336, 448], partial_dims=[48, 72, 96],
                 depths=[4, 7, 6], types=["i", "s", "s"], qk_dims=[16, 16, 16],
                 down_ops=[["subsample", 2], ["subsample", 2], [""]], num_convs: int=2, num_stages: int=3,
                 use_center_att: bool=False, kernel_size: int=3):
        super(SHViT, self).__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dims
        self.partial_dim = partial_dims    # partial_dim = r*embed_dim with r=1/4.67
        self.depth = depths
        self.types = types
        self.qk_dim = qk_dims
        self.down_ops = down_ops
        self.num_stages = num_stages
        self.use_center_att = use_center_att        
        self.kernel_size = kernel_size


        layers = []
        out_channels = in_channels  # Start with input channels
        division_factor = 2 ** (num_convs - 1)  # Adjust division based on num_convs

        # First convolution
        out_channels = embed_dims[0] // division_factor
        layers.append(Conv2d_BN(in_channels=in_channels, out_channels=out_channels, kernel_size=3, strides=2, padding=1))
        layers.append(nn.ReLU())

        # Additional convolutions
        for i in range(num_convs - 1):
            in_channels = out_channels
            out_channels = embed_dims[0] // (division_factor // (2 ** (i + 1)))
            layers.append(Conv2d_BN(in_channels=in_channels, out_channels=out_channels, kernel_size=3, strides=2, padding=1))
            if i != num_convs - 2:  # Don't add ReLU after the last Conv2d_BN
                layers.append(nn.ReLU())

        self.patch_embed = nn.Sequential(*layers)
        
        
        # SHViT Blocks
        self.blocks1, self.blocks2, self.blocks3 = self._build_stage()
        
       
    def _build_stage(self):
        blocks1 = nn.Sequential()
        blocks2 = nn.Sequential()
        blocks3 = nn.Sequential()
       
        for i, (ed, kd, pd, dpth, do, t) in enumerate(zip(self.embed_dim, self.qk_dim, self.partial_dim, self.depth, self.down_ops, self.types)):

            flg_is_last_stage=False     # if last stage, then use SHSA instead of SHCSA       
            if i==len(self.embed_dim)-1:
                flg_is_last_stage=True

            for d in range(dpth):
                eval("blocks" + str(i+1)).append(BasicBlock(dim=ed, qk_dim=kd, pdim=pd, block_type=t, flg_is_last_stage=flg_is_last_stage, 
                                                              use_center_att=self.use_center_att, kernel_size=self.kernel_size))
            if do[0] == "subsample":
                # Build SHViT downsample block
                blk = eval("blocks" + str(i+2))
                blk.add_module("downsample1", nn.Sequential(
                    Residual(Conv2d_BN(in_channels=self.embed_dim[i], out_channels=self.embed_dim[i], kernel_size=3, strides=1, padding=1, groups=self.embed_dim[i])),
                    Residual(FFN(ed=self.embed_dim[i], h=int(self.embed_dim[i] * 2))),
                ))
                blk.add_module("patch_merge", PatchMerging(dim=self.embed_dim[i], out_dim=self.embed_dim[i + 1]))
                blk.add_module("downsample2", nn.Sequential(
                    Residual(Conv2d_BN(in_channels=self.embed_dim[i+1], out_channels=self.embed_dim[i+1], kernel_size=3, strides=1, padding=1, groups=self.embed_dim[i+1])),
                    Residual(FFN(ed=self.embed_dim[i+1], h=int(self.embed_dim[i+1] * 2))),
                ))
       
        return blocks1, blocks2, blocks3
    
 
    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        x = self.patch_embed(inputs)
        x = self.blocks1(x)
        
        if self.num_stages >= 3:
            outs.append(x)
 
        x = self.blocks2(x)
        if self.num_stages >= 2:
            outs.append(x)
 
        x = self.blocks3(x)
        outs.append(x)  # Always append the last output     
        
        return outs