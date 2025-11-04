import torch.nn as nn
import torch.nn.functional as F
from layers import *
from typing import Optional
from timm.models.layers import SqueezeExcite as SEBlock

class PatchEmb(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 128):
        super(PatchEmb, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = self.norm(x)
        return x

class Adb_SE_Transition(nn.Module):
    """
    Este módulo executa a sequência: ADB -> SE -> Transition
    """
    def __init__(self,
                 num_layers: int,
                 in_channels: int,
                 growth_rate: int,
                 bottleneck: bool,
                 p_adb: float = 0.0,
                 theta: float = 0.5,
                 p_transition: float = 0.0,
                 reduction: int = 16):
        
        super(Adb_SE_Transition, self).__init__()

        self.adb = AtrousDenseBlock(
            num_layers=num_layers,
            in_channels=in_channels,
            growth_rate=growth_rate,
            bottleneck=bottleneck,
            p=p_adb
        )
        
        adb_out_channels = self.adb.out_channels
        self.se = SEBlock(channels=adb_out_channels, reduction=reduction)
        
        self.transition = TransitionLayer(
            in_channels=adb_out_channels,
            theta=theta,
            p=p_transition
        )
        self.out_channels = self.transition.out_channels

    def forward(self, x):
        x_adb = self.adb(x)
        x_se = self.se(x_adb)
        x_out = self.transition(x_se)
        
        return x_out

class SwimAD2Net(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(SwimAD2Net, self).__init__()
        self.patch_emb = PatchEmb(img_size=224, patch_size=4, in_chans=3, embed_dim=128)
        self.swin_block1_1 = SwinTransformerBlock(input_channels=128, input_resolution=(56,56), number_of_heads=4, window_size=7, shift_size=0)
        self.swin_block1_2 = SwinTransformerBlock(input_channels=128, input_resolution=(56,56), number_of_heads=4, window_size=7, shift_size=7)
        
        self.atrousdense_block1 = AtrousDenseBlock(num_layers=4, in_channels=128, growth_rate=32, bottleneck=True, p=0.0, dilation_rates=[1,2,4,8])
        
    def forward(self, x): # x: [B, 3, 224, 224]
        x = self.patch_emb(x)  # [B, 128, 56, 56]
        x = self.swin_block1_1(x) # [B, 128, 56, 56]
        x = self.swin_block1_2(x) # [B, 128, 56, 56]
        
        return x
