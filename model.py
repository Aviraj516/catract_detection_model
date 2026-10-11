import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                channels // reduction,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU(),

            nn.Conv2d(
                channels // reduction,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = self.mlp(self.avg_pool(x))
        maximum = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg + maximum)

        return x * attention


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)
        maximum, _ = torch.max(x, dim=1, keepdim=True)

        combined = torch.cat([avg, maximum], dim=1)

        attention = self.sigmoid(self.conv(combined))

        return x * attention


class CBAM(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class CataractModel(nn.Module):

    def __init__(self):

        super().__init__()

        weights = EfficientNet_V2_S_Weights.DEFAULT

        self.backbone = efficientnet_v2_s(weights=weights)

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # EfficientNetV2-S final feature size
        self.feature_dim = 1280

        self.attention = CBAM(self.feature_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(
                self.feature_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                1
            )
        )

    def forward(self, x):

        # Extract convolutional features
        x = self.backbone.features(x)

        # Attention
        x = self.attention(x)

        # Global pooling
        x = self.pool(x)

        # Classification
        x = self.classifier(x)

        return x