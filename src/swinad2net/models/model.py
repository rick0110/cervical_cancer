import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
from typing import Optional, List, Tuple, Dict
import random


class SqueezeExcitation(nn.Module):
    """Channel attention block used in A2SDNet121."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class A2SDDenseLayer(nn.Module):
    """Dense layer with optional atrous 3x3 convolution."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            inter_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.drop_rate = drop_rate

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(features, dim=1)
        x = self.conv1(self.relu1(self.norm1(x)))
        x = self.conv2(self.relu2(self.norm2(x)))

        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


class A2SDAtrousDenseBlock(nn.Module):
    """
    Atrous Dense Block (ADB) used in A2SDNet121.

    The last three dense layers use atrous convolutions with dilation rates
    1, 2 and 3, matching the paper's design.
    """

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        if num_layers >= 3:
            dilations = [1] * (num_layers - 3) + [1, 2, 3]
        else:
            dilations = [1] * num_layers

        channels = in_channels
        for dilation in dilations:
            layer = A2SDDenseLayer(
                in_channels=channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                dilation=dilation,
            )
            self.layers.append(layer)
            channels += growth_rate

        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(features))
        return torch.cat(features, dim=1)


class A2SDTransition(nn.Module):
    """Transition layer used by A2SDNet121."""

    def __init__(self, in_channels: int, compression: float = 0.5) -> None:
        super().__init__()
        out_channels = int(in_channels * compression)
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

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
                dilation_rates: Optional[List[int]] = [1, 2, 3]):
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
            - self.num_classes: Number of output classes.
        """
        super(SwinAD2Net, self).__init__()

        self.num_classes = num_classes
        
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


class Densenet121(nn.Module):
    """
    DenseNet-121 backbone as described in Zhang et al. (2025) with the standard
    bottleneck layout and transition layers.
    """

    def __init__(self,
                 num_classes: int = 2,
                 growth_rate: int = 32,
                 block_config: Tuple[int, ...] = (6, 12, 24, 16),
                 bn_size: int = 4,
                 drop_rate: float = 0.0,
                 num_init_features: int = 64,
                 compression: float = 0.5,
                 in_channels: int = 3) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Stem: 7x7 conv + BN + ReLU + 3x3 max pool
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers,
                               in_channels=num_features,
                               growth_rate=growth_rate,
                               bn_size=bn_size,
                               drop_rate=drop_rate)
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = block.out_channels

            if i != len(block_config) - 1:
                transition = TransitionLayer(in_channels=num_features,
                                             theta=compression,
                                             p=drop_rate)
                self.features.add_module(f"transition{i + 1}", transition)
                num_features = transition.out_channels

        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        out = self.classifier(out)
        return out


class A2SDNet121(nn.Module):
    """
    A2SDNet121 from Zhang et al. (Scientific Reports, 2025).

    Key characteristics replicated from the paper:
    - Optimized Stem layer: 3x3 conv (s=1, p=1) + 2x2 max-pool (s=2, p=0)
    - Four Atrous Dense Blocks with depths (6, 12, 16, 24)
    - Last three layers in each ADB use atrous rates (1, 2, 3)
    - SE attention after each ADB
    - Transition layers between the first three ADB modules
    """

    def __init__(
        self,
        num_classes: int = 2,
        growth_rate: int = 32,
        block_config: Tuple[int, ...] = (6, 12, 16, 24),
        bn_size: int = 4,
        drop_rate: float = 0.0,
        num_init_features: int = 64,
        compression: float = 0.5,
        in_channels: int = 3,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Stem layer as described in the paper.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.blocks = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        num_features = num_init_features
        for idx, num_layers in enumerate(block_config):
            block = A2SDAtrousDenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.blocks.append(block)
            num_features = block.out_channels

            self.se_blocks.append(SqueezeExcitation(num_features, reduction=se_reduction))

            if idx != len(block_config) - 1:
                transition = A2SDTransition(num_features, compression=compression)
                self.transitions.append(transition)
                num_features = transition.out_channels

        self.norm_final = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            x = self.se_blocks[idx](x)
            if idx < len(self.transitions):
                x = self.transitions[idx](x)

        x = F.relu(self.norm_final(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.classifier(x)

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
                dilation_rates: Optional[List[int]] = [1, 2, 3],
                compression_rates: Optional[List[float]] = [0.25, 0.25, 0.25],
                dropout: float = 0.5,
                drop_path: float = 0.1
                ):
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
            - self.num_classes: Number of output classes.
            - self.patch_emb: Patch embedding layer.
            - self.swin_blockX_Y: Swin Transformer blocks for each stage.
            - self.adb_se_transX: Atrous Dense Block (ASPP-like) with SE and Transition for each stage.
            - self.global_pool: Global average pooling layer.
            - self.classifier: Fully connected layer for classification.

        """
        super(SwinAD2Net_ASPP_like, self).__init__()

        self.num_classes = num_classes
        self.drop_path = drop_path
        self.dropout = dropout
        
        # Patch Embedding
        self.patch_emb = PatchEmb(patch_size=patch_size_embed, in_channels=3, embed_dim=embed_dim)
        
        # Stage 1: 2x Swin Blocks (56x56)
        self.swin_block1_1 = SwinTransformerBlock(in_channels=embed_dim, input_resolution=(image_size//patch_size_embed, image_size//patch_size_embed), number_of_heads=4, window_size=7, shift_size=0, dropout=dropout)
        self.swin_block1_2 = SwinTransformerBlock(in_channels=embed_dim, input_resolution=(image_size//patch_size_embed, image_size//patch_size_embed), number_of_heads=4, window_size=7, shift_size=3, dropout=dropout)
        # ADB -> SE -> Transition 1
        self.adb_se_trans1 = Adb_SE_Transition_ASPP_like(in_channels=embed_dim, growth_rate=growth_rate, theta=0.5, dilation_rates=dilation_rates, compression_rate=compression_rates[0], p_transition=dropout)

        # Stage 2: 2x Swin Blocks (28x28)
        ch2 = self.adb_se_trans1.out_channels
        self.swin_block2_1 = SwinTransformerBlock(in_channels=ch2, input_resolution=(image_size//(patch_size_embed*2), image_size//(patch_size_embed*2)), number_of_heads=4, window_size=7, shift_size=0, dropout=dropout)
        self.swin_block2_2 = SwinTransformerBlock(in_channels=ch2, input_resolution=(image_size//(patch_size_embed*2), image_size//(patch_size_embed*2)), number_of_heads=4, window_size=7, shift_size=3, dropout=dropout)

        # ADB -> SE -> Transition 2
        self.adb_se_trans2 = Adb_SE_Transition_ASPP_like(in_channels=ch2, growth_rate=growth_rate, theta=0.5, dilation_rates=dilation_rates, compression_rate=compression_rates[1], p_transition=dropout)
        # Stage 3: 6x Swin Blocks (14x14)
        ch3 = self.adb_se_trans2.out_channels
        self.swin_block3_1 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=0, dropout=dropout)
        self.swin_block3_2 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=3, dropout=dropout)
        self.swin_block3_3 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=0, dropout=dropout)
        self.swin_block3_4 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=3, dropout=dropout)
        self.swin_block3_5 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=0, dropout=dropout)
        self.swin_block3_6 = SwinTransformerBlock(in_channels=ch3, input_resolution=(image_size//(patch_size_embed*4), image_size//(patch_size_embed*4)), number_of_heads=2, window_size=7, shift_size=3, dropout=dropout)
        
        # ADB -> SE -> Transition 3
        self.adb_se_trans3 = Adb_SE_Transition_ASPP_like(in_channels=ch3, growth_rate=growth_rate, theta=0.5, dilation_rates=dilation_rates, compression_rate=compression_rates[2], p_transition=dropout)
        
        # Stage 4: 2x Swin Blocks (7x7)
        ch4 = self.adb_se_trans3.out_channels
        self.swin_block4_1 = SwinTransformerBlock(in_channels=ch4, input_resolution=(image_size//(patch_size_embed*8), image_size//(patch_size_embed*8)), number_of_heads=1, window_size=7, shift_size=0, dropout=dropout)
        self.swin_block4_2 = SwinTransformerBlock(in_channels=ch4, input_resolution=(image_size//(patch_size_embed*8), image_size//(patch_size_embed*8)), number_of_heads=1, window_size=7, shift_size=3, dropout=dropout)

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
        if self.training and random.random() < self.drop_path:
            pass
        else:
            x = self.swin_block3_1(x)  # [B, ch3, 14, 14]
            x = self.swin_block3_2(x)  # [B, ch3, 14, 14]

        if self.training and random.random() < self.drop_path:
            pass
        else:
            x = self.swin_block3_3(x)  # [B, ch3, 14, 14]
            x = self.swin_block3_4(x)  # [B, ch3, 14, 14]
        
        if self.training and random.random() < self.drop_path:
            pass
        else:
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



