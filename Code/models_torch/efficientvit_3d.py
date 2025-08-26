# slightly adapted from: https://github.com/microsoft/Cream/blob/main/EfficientViT/downstream/efficientvit.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
import itertools


# from timm.models.vision_transformer import trunc_normal_
# from timm.models.layers import SqueezeExcite, DropPath, to_2tuple

import numpy as np
import itertools

# from mmcv_custom import load_checkpoint, _load_checkpoint, load_state_dict
# from mmdet.utils import get_root_logger
# from mmdet.models.builder import BACKBONES
# from torch.nn.modules.batchnorm import _BatchNorm



import torch
import torch.nn as nn
import itertools

class Conv3d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()

        self.add_module('c', torch.nn.Conv3d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm3d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5  
        w = c.weight * w[:, None, None, None, None]  # (out, in, kD, kH, kW)
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv3d(w.size(1) * c.groups, w.size(0), 
                            w.shape[2:], stride=c.stride, 
                            padding=c.padding, dilation=c.dilation, 
                            groups=c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class CascadedGroupAttention3D(torch.nn.Module):
    """
    Cascaded Group Attention in 3D (B, C, D, H, W).
    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key per head.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (tuple or int): Input resolution (D,H,W) or single int to be used for all dims.
        kernels (List[int] or List[tuple]): Kernel sizes for the depthwise conv per head. Each element can be int ( -> (k,k,k)) or 3-tuple.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio
        
        qkvs = []
        dws = []
        for i in range(num_heads):
            # per-head qkv conv (1x1x1)
            # qkvs.append(Conv3d_BN(dim // (num_heads), self.key_dim * 2 + self.d, ks=1))
            # depthwise conv on q: in_channels = key_dim, out_channels = key_dim, groups=key_dim
        #     ksize = kernels[i]
        #     pad = (ksize[0]//2, ksize[1]//2, ksize[2]//2)
        #     dws.append(Conv3d_BN(self.key_dim, self.key_dim, ks=ksize, stride=1, pad=pad, groups=self.key_dim))
        # self.qkvs = torch.nn.ModuleList(qkvs)
        # self.dws = torch.nn.ModuleList(dws)
        # self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv3d_BN(
        #     self.d * num_heads, dim, ks=1, bn_weight_init=0))

            qkvs.append(Conv3d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv3d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv3d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        

        points = list(itertools.product(range(resolution), range(resolution), range(resolution)))
        # points = list(itertools.product(range(resolution), range(resolution)))

        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]), abs(p1[2] - p2[2]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        # attention_biases: one bias per head x number of unique offsets
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))  # N x N


    

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            # cache expanded biases for inference to save indexing cost
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    
    def forward(self, x):  # x (B,C,D,H,W)
        B, C, D, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]  # (num_heads, N, N)
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
            
        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # add the previous output to the input (residual across head groups)
                feat = feat + feats_in[i]
            feat = qkv(feat)  # produces B, (2*key_dim + d), D, H, W
            # split into q,k,v
            q, k, v = feat.view(B, -1, D, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C_each, D, H, W
            q = self.dws[i](q)  # depthwise conv on q: B, key_dim, D, H, W
            # flatten spatial dims to N = D*H*W
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C_each, N
            # attention: (q^T @ k) -> (B, N, N) assuming q and k have same channel dim (key_dim)
            
            ######################################################################
            
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )

            ######################################################################
            
            attn = attn.softmax(dim=-1)  # B, N, N
            # output: (v @ attn^T) -> (B, d, N)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, D, H, W)  # B, d, D, H, W
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))  # B, C, D, H, W
        return x


# -----------------------------
# Kurzes Beispiel / Testlauf
# -----------------------------
if __name__ == "__main__":
    '''
    # Beispiel: B=2, C=64, D=8, H=16, W=16
    B, C, D, H, W = 2, 64, 8, 16, 16
    x = torch.randn(B, C, D, H, W)

    # Parameter: dim must be divisible durch num_heads
    dim = C
    key_dim = 8
    num_heads = 8
    attn_ratio = 4
    # Achtung: D*H*W kann schnell groß werden — nutze moderate Größen beim Testen
    module = CascadedGroupAttention3D(dim=dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, resolution=(D,H,W), kernels=[3]*num_heads)
    out = module(x)  # out: (B, C, D, H, W)
    print("Output shape:", out.shape)
    '''

    x = torch.randn(2, 64, 14, 14, 14)

    model = CascadedGroupAttention3D(
        dim=64,
        key_dim=8,        # <-- 8
        num_heads=4,
        attn_ratio=2,     # d = 2*8 = 16 == 64//4
        resolution=14,
        kernels=[3, 3, 3, 3]
    )

    y = model(x)

    print("====="*10)
    print("Input shape:", x.shape) # Input shape: torch.Size([2, 64, 14, 14, 14])
    print("====="*10)
    print("Output shape:", y.shape) # Output shape: torch.Size([2, 64, 14, 14, 14])
    print("====="*10)