import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)


class QualityModel(nn.Module):

    def __init__(self):

        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT

        self.backbone = mobilenet_v3_small(weights=weights)

        # Replace final classifier
        in_features = self.backbone.classifier[-1].in_features

        self.backbone.classifier[-1] = nn.Linear(
            in_features,
            1
        )

    def forward(self, x):

        return self.backbone(x)