# from: https://github.com/IMvision12/SegFormer-tf/blob/main/models/segformer.py
# converted to pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_torch.modules import MixVisionTransformer
from models_torch.head import SegFormerHead
from models_torch.utils import ResizeLayer
from models_torch.shvit import SHViT


MODEL_CONFIGS = {
    "mit_b0": {
        "embed_dims": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "decode_dim": 256,
    },
    "mit_b1": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "decode_dim": 256,
    },
    "mit_b2": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "decode_dim": 768,
    },
    "mit_b3": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "decode_dim": 768,
    },
    "mit_b4": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 8, 27, 3],
        "decode_dim": 768,
    },
    "mit_b5": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 6, 40, 3],
        "decode_dim": 768,
    },
}


# from: https://github.com/ysj9909/SHViT/blob/main/model/build.py
SHVIT_CONFIGS = {
    "SHViT_s1": {
        "embed_dims": [128, 224, 320],
        "partial_dims": [32, 48, 68],
        "depths": [2, 4, 5],
        "types": ["i", "s", "s"],
    },
    "SHViT_s2": {
        "embed_dims": [128, 308, 448],
        "partial_dims": [32, 66, 96],
        "depths": [2, 4, 5],
        "types": ["i", "s", "s"],
    },
    "SHViT_s3": {
        "embed_dims": [192, 352, 448],
        "partial_dims": [48, 75, 96],
        "depths": [3, 5, 5],
        "types": ["i", "s", "s"],
    },
    "SHViT_s4": {
        "embed_dims": [224, 336, 448],
        "partial_dims": [48, 72, 96],
        "depths": [4, 7, 6],
        "types": ["i", "s", "s"],
    },
}


class SegFormer(nn.Module):
    """
    SegFormer2D: A 2D segmentation model leveraging MixVisionTransformer, SegFormerHead, and optional resizing.

    Args:
        model_type (str): Type of the model (e.g., "B0", "B1", ...).
        input_shape (tuple of int): Shape of the input tensor (C, H, W), where C is the number of input channels,
                                    H is the height, and W is the width.
        num_classes (int, optional): Number of output classes for segmentation. Defaults to 7.
        use_resize (bool, optional): Whether to resize the output to the input shape. Defaults to True.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, C is the number of input channels,
                          H is the height, and W is the width.

    Outputs:
        torch.Tensor: Output tensor of shape (B, num_classes, H, W) after applying segmentation and optional resizing.
    """    
    def __init__(self, model_type: str="B0", shvit_type: str="", input_shape: tuple=(3,224,224), 
                 num_convs: int=2, num_stages: int=3, num_classes: int=7, use_center_att: bool=False, use_resize: bool=True):
        super(SegFormer, self).__init__()
        # 'shvit_type', 'num_convs', 'use_center_att' , and 'num_stages' are included for compatibility with class SegFormer3D_SHViT,
        # but are not used in this implementation!
        assert len(input_shape) == 3, "input_shape must be a tuple of length 3 (C, H, W)"
        assert input_shape[1] == input_shape[2], "H, and W dimensions must be equal"
        model_type = model_type.lower()
        self.use_resize = use_resize

        self.mix_vision_transformer = MixVisionTransformer(
            img_size=input_shape[1], img_channels=input_shape[0],
            embed_dims=MODEL_CONFIGS[f"mit_{model_type}"]["embed_dims"],
            depths=MODEL_CONFIGS[f"mit_{model_type}"]["depths"]
        )

        self.seg_former_head = SegFormerHead(
            num_classes=num_classes,
            decode_dim=MODEL_CONFIGS[f"mit_{model_type}"]["decode_dim"],
            input_dims=MODEL_CONFIGS[f"mit_{model_type}"]["embed_dims"]
        )

        self.resize_layer = ResizeLayer(input_shape[1], input_shape[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mix_vision_transformer(x)
        x = self.seg_former_head(x)

        if self.use_resize:
            x = self.resize_layer(x)

        return F.softmax(x, dim=1).to(torch.float32)
    
        
    
class SegFormer_SHViT(nn.Module):   
    """
    SegFormer_SHViT: A 2D segmentation model leveraging SHViT, SegFormerHead, and optional resizing.
 
    Args:
        model_type (str): Type of the model (e.g., "B0").
        shvit_type (str): Type of the shvit config (e.g., "S4").
        input_shape (tuple of int): Shape of the input tensor (C, H, W), where C is the number of input channels,
                                    H is the height, and W is the width.
        num_convs (int, optional): Number of conv2d_bn layers in head. Defaults to 2.
        num_stages (int, optional): Number of stages (output from shvit). Defaults to 3.
        num_classes (int, optional): Number of output classes for segmentation. Defaults to 7.
        use_center_att (bool, optional): Whether to use center_attention in 2nd stage. Defaults to False.
        use_resize (bool, optional): Whether to resize the output to the input shape. Defaults to True.
        kernel_size (int, default=3): Size of the convolutional kernel used in attention.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, C is the number of input channels,
                          H is the height, and W is the width.
 
    Outputs:
        torch.Tensor: Output tensor of shape (B, num_classes, H, W) after applying segmentation and optional resizing.
    """
    def __init__(self, model_type: str="B0", shvit_type: str="S4", input_shape: tuple=(3,224,224),
                 num_convs: int=2, num_stages: int=3, num_classes: int=7, use_center_att: bool=False, use_resize: bool=True, kernel_size: int=3):
        super(SegFormer_SHViT, self).__init__()
        
        assert len(input_shape) == 3, "input_shape must be a tuple of length 3 (C, H, W)"
        assert input_shape[1] == input_shape[2], "H, and W dimensions must be equal"
        assert 1 <= num_convs <= 4, "num_convs should be between 1 and 4"
    
        model_type = model_type.lower()
        shvit_type = shvit_type.lower()
        self.use_resize = use_resize
 
        self.shvit = SHViT(
            in_channels=input_shape[0],
            embed_dims=SHVIT_CONFIGS[f"SHViT_{shvit_type}"]["embed_dims"],
            partial_dims=SHVIT_CONFIGS[f"SHViT_{shvit_type}"]["partial_dims"],
            depths=SHVIT_CONFIGS[f"SHViT_{shvit_type}"]["depths"],
            types=SHVIT_CONFIGS[f"SHViT_{shvit_type}"]["types"],
            num_convs=num_convs,
            num_stages=num_stages,
            use_center_att=use_center_att,
            kernel_size=kernel_size
        )
       
        self.seg_former_head = SegFormerHead(
            num_classes=num_classes,
            input_dims=SHVIT_CONFIGS[f"SHViT_{shvit_type}"]["embed_dims"][-num_stages:],
            decode_dim=MODEL_CONFIGS[f"mit_{model_type}"]["decode_dim"],
        )
 
        self.resize_layer = ResizeLayer(input_shape[1], input_shape[2])
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shvit(x)
        x = self.seg_former_head(x)
 
        if self.use_resize:
            x = self.resize_layer(x)
 
        return F.softmax(x, dim=1).to(torch.float32)   