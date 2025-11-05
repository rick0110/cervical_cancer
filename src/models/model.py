import torch.nn as nn
import torch.nn.functional as F
from layers import *
from typing import Optional
from timm.models.layers import SqueezeExcite as SEBlock
from layers import AtrousDenseBlock, TransitionLayer, SwinTransformerBlock

class PatchEmb(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 128):
        super(PatchEmb, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        # LayerNorm espera [B, H, W, C], então permute
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
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
            in_channels=in_channels,
            growth_rate=growth_rate,
        )
        
        adb_out_channels = self.adb.out_channels
        self.se = SEBlock(channels=adb_out_channels)
        
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

class SwinAD2Net(nn.Module):
    def __init__(self, num_classes: int = 2, image_size: int = 224):
        super(SwinAD2Net, self).__init__()
        
        # Patch Embedding
        self.patch_emb = PatchEmb(img_size=image_size, patch_size=4, in_chans=3, embed_dim=256)
        
        # Stage 1: 2x Swin Blocks (56x56)
        self.swin_block1_1 = SwinTransformerBlock(in_channels=256, input_resolution=(56, 56), number_of_heads=4, window_size=7, shift_size=0)
        self.swin_block1_2 = SwinTransformerBlock(in_channels=256, input_resolution=(56, 56), number_of_heads=4, window_size=7, shift_size=3)

        # ADB -> SE -> Transition 1
        self.adb_se_trans1 = Adb_SE_Transition(num_layers=4, in_channels=256, growth_rate=32, bottleneck=True, theta=0.5, reduction=16)

        # Stage 2: 2x Swin Blocks (28x28)
        ch2 = self.adb_se_trans1.out_channels
        self.swin_block2_1 = SwinTransformerBlock(in_channels=ch2, input_resolution=(28, 28), number_of_heads=4, window_size=7, shift_size=0)
        self.swin_block2_2 = SwinTransformerBlock(in_channels=ch2, input_resolution=(28, 28), number_of_heads=4, window_size=7, shift_size=3)

        # ADB -> SE -> Transition 2
        self.adb_se_trans2 = Adb_SE_Transition(num_layers=4, in_channels=ch2, growth_rate=32, bottleneck=True, theta=0.5, reduction=16)
        
        # Stage 3: 6x Swin Blocks (14x14)
        ch3 = self.adb_se_trans2.out_channels
        self.swin_block3_1 = SwinTransformerBlock(in_channels=ch3, input_resolution=(14, 14), number_of_heads=2, window_size=7, shift_size=0)
        self.swin_block3_2 = SwinTransformerBlock(in_channels=ch3, input_resolution=(14, 14), number_of_heads=2, window_size=7, shift_size=3)
        self.swin_block3_3 = SwinTransformerBlock(in_channels=ch3, input_resolution=(14, 14), number_of_heads=2, window_size=7, shift_size=0)
        self.swin_block3_4 = SwinTransformerBlock(in_channels=ch3, input_resolution=(14, 14), number_of_heads=2, window_size=7, shift_size=3)
        self.swin_block3_5 = SwinTransformerBlock(in_channels=ch3, input_resolution=(14, 14), number_of_heads=2, window_size=7, shift_size=0)
        self.swin_block3_6 = SwinTransformerBlock(in_channels=ch3, input_resolution=(14, 14), number_of_heads=2, window_size=7, shift_size=3)
        
        # ADB -> SE -> Transition 3
        self.adb_se_trans3 = Adb_SE_Transition(num_layers=4, in_channels=ch3, growth_rate=32, bottleneck=True, theta=0.5, reduction=16)
        
        # Stage 4: 2x Swin Blocks (7x7)
        ch4 = self.adb_se_trans3.out_channels
        self.swin_block4_1 = SwinTransformerBlock(in_channels=ch4, input_resolution=(7, 7), number_of_heads=1, window_size=7, shift_size=0)
        self.swin_block4_2 = SwinTransformerBlock(in_channels=ch4, input_resolution=(7, 7), number_of_heads=1, window_size=7, shift_size=3)

        # Global Average Pooling + Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch4, num_classes)
    
    def forward(self, x):
        # x: [B, 3, 224, 224]
        
        # Patch Embedding
        x = self.patch_emb(x)  # [B, 128, 56, 56]
        
        # Stage 1: 2x Swin Blocks
        x = self.swin_block1_1(x)  # [B, 128, 56, 56]
        x = self.swin_block1_2(x)  # [B, 128, 56, 56]
        
        # ADB -> SE -> Transition 1
        x = self.adb_se_trans1(x)  # [B, ch2, 28, 28]
        
        # Stage 2: 2x Swin Blocks
        x = self.swin_block2_1(x)  # [B, ch2, 28, 28]
        x = self.swin_block2_2(x)  # [B, ch2, 28, 28]
        
        # ADB -> SE -> Transition 2
        x = self.adb_se_trans2(x)  # [B, ch3, 14, 14]
        
        # Stage 3: 6x Swin Blocks
        x = self.swin_block3_1(x)  # [B, ch3, 14, 14]
        x = self.swin_block3_2(x)  # [B, ch3, 14, 14]
        x = self.swin_block3_3(x)  # [B, ch3, 14, 14]
        x = self.swin_block3_4(x)  # [B, ch3, 14, 14]
        x = self.swin_block3_5(x)  # [B, ch3, 14, 14]
        x = self.swin_block3_6(x)  # [B, ch3, 14, 14]
        
        # ADB -> SE -> Transition 3
        x = self.adb_se_trans3(x)  # [B, ch4, 7, 7]
        
        # Stage 4: 2x Swin Blocks
        x = self.swin_block4_1(x)  # [B, ch4, 7, 7]
        x = self.swin_block4_2(x)  # [B, ch4, 7, 7]
        
        # Global Pooling + Classifier
        x = self.global_pool(x)  # [B, ch4, 1, 1]
        x = x.view(x.size(0), -1)  # [B, ch4]
        x = self.classifier(x)  # [B, num_classes]
        
        return x
