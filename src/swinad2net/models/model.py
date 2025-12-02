import torch.nn as nn
import torch.nn.functional as F
from .layers import *
from typing import Optional, List

class SwinAD2Net(nn.Module):
    """
    SwinAD2Net model combining Swin Transformer blocks with Atrous Dense Blocks and SE transitions.
    The architecture consists of multiple stages of Swin Transformer blocks followed by Atrous Dense Blocks
    with Squeeze-and-Excitation and Transition layers for downsampling.
    Finally, a global average pooling and a fully connected layer are used for classification.
    """
    def __init__(self, num_classes: int = 2,
                embed_dim: int = 128,
                image_size: int = 224,
                patch_size_embed: int = 4,
                growth_rate: int = 32,
                dilation_rates: list = [1, 2, 3]):
        """Initialize the SwinAD2Net model.
        
        Args:
            - num_classes (int): Number of output classes for classification.
            - embed_dim (int): Dimension of the embedding after patch embedding.
            - image_size (int): Input image size (assumed square).
            - patch_size_embed (int): Patch size for the patch embedding layer.
            - growth_rate (int): Growth rate for the Atrous Dense Blocks.
            - dilation_rates (list): List of dilation rates for the Atrous Dense Blocks.

        Params
        ------
            - self.patch_emb: Patch embedding layer.
            - self.swin_blockX_Y: Swin Transformer blocks for each stage.
            - self.adb_se_transX: Atrous Dense Block with SE and Transition for each stage.
            - self.global_pool: Global average pooling layer.
            - self.classifier: Fully connected layer for classification.
        """
        super(SwinAD2Net, self).__init__()
        
        # Patch Embedding
        self.patch_emb = PatchEmb(patch_size=patch_size_embed, in_channels=3, embed_dim=embed_dim)
        
        # Stage 1: 2x Swin Blocks (56x56)
        self.swin_block1_1 = SwinTransformerBlock(in_channels=embed_dim, input_resolution=(image_size//patch_size_embed, image_size//patch_size_embed), number_of_heads=4, window_size=7, shift_size=0)
        self.swin_block1_2 = SwinTransformerBlock(in_channels=embed_dim, input_resolution=(image_size//patch_size_embed, image_size//patch_size_embed), number_of_heads=4, window_size=7, shift_size=3)
        # ADB -> SE -> Transition 1
        self.adb_se_trans1 = Adb_SE_Transition(in_channels=embed_dim, growth_rate=growth_rate, theta=0.5, dilation_rates=dilation_rates)

        # Stage 2: 2x Swin Blocks (28x28)
        ch2 = self.adb_se_trans1.out_channels
        self.swin_block2_1 = SwinTransformerBlock(in_channels=ch2, input_resolution=(image_size//(patch_size_embed*2), image_size//(patch_size_embed*2)), number_of_heads=4, window_size=7, shift_size=0)
        self.swin_block2_2 = SwinTransformerBlock(in_channels=ch2, input_resolution=(image_size//(patch_size_embed*2), image_size//(patch_size_embed*2)), number_of_heads=4, window_size=7, shift_size=3)

        # ADB -> SE -> Transition 2
        self.adb_se_trans2 = Adb_SE_Transition(in_channels=ch2, growth_rate=growth_rate, theta=0.5, dilation_rates=dilation_rates)

        # Stage 3: 6x Swin Blocks (14x14)
        ch3 = self.adb_se_trans2.out_channels
        self.swin_block3_1 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=0)
        self.swin_block3_2 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=3)
        self.swin_block3_3 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=0)
        self.swin_block3_4 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=3)
        self.swin_block3_5 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=0)
        self.swin_block3_6 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=3)
        
        # ADB -> SE -> Transition 3
        self.adb_se_trans3 = Adb_SE_Transition(in_channels=ch3, growth_rate=growth_rate, theta=0.5, dilation_rates=dilation_rates)
        
        # Stage 4: 2x Swin Blocks (7x7)
        ch4 = self.adb_se_trans3.out_channels
        self.swin_block4_1 = SwinTransformerBlock(in_channels=ch4, input_resolution=(image_size//(patch_size_embed*8), image_size//(patch_size_embed*8)), number_of_heads=1, window_size=7, shift_size=0)
        self.swin_block4_2 = SwinTransformerBlock(in_channels=ch4, input_resolution=(image_size//(patch_size_embed*8), image_size//(patch_size_embed*8)), number_of_heads=1, window_size=7, shift_size=3)

        # Global Average Pooling + Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch4, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the SwinAD2Net model.
        """
        # x: [B, 3, 224, 224]
        
        # Patch Embedding
        x = self.patch_emb(x)  # [B, embed_dim, 56, 56]
        
        # Stage 1: 2x Swin Blocks
        x = self.swin_block1_1(x)  # [B, embed_dim, 56, 56]
        x = self.swin_block1_2(x)  # [B, embed_dim, 56, 56]
        
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

class SwinAD2Net_ASPP_like(nn.Module):
    """
    SwinAD2Net model combining Swin Transformer blocks with Atrous Dense Blocks (ASPP-like) and SE transitions.
    The architecture consists of multiple stages of Swin Transformer blocks followed by Atrous Dense Blocks
    with Squeeze-and-Excitation and Transition layers for downsampling.
    Finally, a global average pooling and a fully connected layer are used for classification."""
    def __init__(self, num_classes: int = 2,
                embed_dim: int = 128,
                image_size: int = 224,
                patch_size_embed: int = 4,
                growth_rate: int = 32,
                dilation_rates: list = [1, 2, 3]):
        """
        Initialize the SwinAD2Net_ASPP_like model.

        Args:
            - num_classes (int): Number of output classes for classification.
            - embed_dim (int): Dimension of the embedding after patch embedding.
            - image_size (int): Input image size (assumed square).
            - patch_size_embed (int): Patch size for the patch embedding layer.
            - growth_rate (int): Growth rate for the Atrous Dense Blocks.
            - dilation_rates (list): List of dilation rates for the Atrous Dense Blocks.

        Params
        ------
            - self.patch_emb: Patch embedding layer.
            - self.swin_blockX_Y: Swin Transformer blocks for each stage.
            - self.adb_se_transX: Atrous Dense Block (ASPP-like) with SE and Transition for each stage.
            - self.global_pool: Global average pooling layer.
            - self.classifier: Fully connected layer for classification.

        """
        super(SwinAD2Net_ASPP_like, self).__init__()
        
        # Patch Embedding
        self.patch_emb = PatchEmb(patch_size=patch_size_embed, in_channels=3, embed_dim=embed_dim)
        
        # Stage 1: 2x Swin Blocks (56x56)
        self.swin_block1_1 = SwinTransformerBlock(in_channels=embed_dim, input_resolution=(image_size//patch_size_embed, image_size//patch_size_embed), number_of_heads=4, window_size=7, shift_size=0)
        self.swin_block1_2 = SwinTransformerBlock(in_channels=embed_dim, input_resolution=(image_size//patch_size_embed, image_size//patch_size_embed), number_of_heads=4, window_size=7, shift_size=3)
        # ADB -> SE -> Transition 1
        self.adb_se_trans1 = Adb_SE_Transition_ASPP_like(in_channels=embed_dim, growth_rate=growth_rate, theta=0.5, dilation_rates=dilation_rates)

        # Stage 2: 2x Swin Blocks (28x28)
        ch2 = self.adb_se_trans1.out_channels
        self.swin_block2_1 = SwinTransformerBlock(in_channels=ch2, input_resolution=(image_size//(patch_size_embed*2), image_size//(patch_size_embed*2)), number_of_heads=4, window_size=7, shift_size=0)
        self.swin_block2_2 = SwinTransformerBlock(in_channels=ch2, input_resolution=(image_size//(patch_size_embed*2), image_size//(patch_size_embed*2)), number_of_heads=4, window_size=7, shift_size=3)

        # ADB -> SE -> Transition 2
        self.adb_se_trans2 = Adb_SE_Transition_ASPP_like(in_channels=ch2, growth_rate=growth_rate, theta=0.5, dilation_rates=dilation_rates)

        # Stage 3: 6x Swin Blocks (14x14)
        ch3 = self.adb_se_trans2.out_channels
        self.swin_block3_1 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=0)
        self.swin_block3_2 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=3)
        self.swin_block3_3 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=0)
        self.swin_block3_4 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=3)
        self.swin_block3_5 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=0)
        self.swin_block3_6 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=3)
        
        # ADB -> SE -> Transition 3
        self.adb_se_trans3 = Adb_SE_Transition_ASPP_like(in_channels=ch3, growth_rate=growth_rate, theta=0.5, dilation_rates=dilation_rates)
        
        # Stage 4: 2x Swin Blocks (7x7)
        ch4 = self.adb_se_trans3.out_channels
        self.swin_block4_1 = SwinTransformerBlock(in_channels=ch4, input_resolution=(image_size//(patch_size_embed*8), image_size//(patch_size_embed*8)), number_of_heads=1, window_size=7, shift_size=0)
        self.swin_block4_2 = SwinTransformerBlock(in_channels=ch4, input_resolution=(image_size//(patch_size_embed*8), image_size//(patch_size_embed*8)), number_of_heads=1, window_size=7, shift_size=3)

        # Global Average Pooling + Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch4, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the SwinAD2Net_ASPP_like model.
        """
        # x: [B, 3, 224, 224]
        
        # Patch Embedding
        x = self.patch_emb(x)  # [B, embed_dim, 56, 56]
        
        # Stage 1: 2x Swin Blocks
        x = self.swin_block1_1(x)  # [B, embed_dim, 56, 56]
        x = self.swin_block1_2(x)  # [B, embed_dim, 56, 56]
        
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
