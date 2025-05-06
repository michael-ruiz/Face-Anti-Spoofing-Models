import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)),
            ('norm', nn.BatchNorm2d(out_channels)),
            ('act', nn.SiLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)

class MobileViTv3_XS(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # First conv layer
        self.conv_1 = ConvBNAct(3, 16, 3, 2, 1)
        
        # Configuration for each stage
        configs = [
            [(16, 32, 2), (32, 32, 1)],           # Stage 1
            [(32, 48, 2), (48, 48, 1), (48, 48, 1)],  # Stage 2
            [(48, 96, 2), (96, 96, 1), (96, 96, 1)],  # Stage 3
            [(96, 160, 2), (160, 160, 1)],        # Stage 4
            [(160, 160, 1)]                        # Stage 5
        ]
        
        # Build layers
        self.layers = nn.ModuleList()
        for stage_id, stage_config in enumerate(configs):
            stage = []
            for in_ch, out_ch, stride in stage_config:
                stage.append(self._make_layer(in_ch, out_ch, stride))
            self.layers.append(nn.Sequential(*stage))
        
        # Final expansion and classifier
        self.conv_1x1_exp = ConvBNAct(160, 640, 1)
        self.classifier = nn.Sequential(OrderedDict([
            ('global_pool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten()),
            ('fc', nn.Linear(640, num_classes))
        ]))

    def _make_layer(self, in_channels, out_channels, stride=1):
        exp_channels = in_channels * 4
        return nn.Sequential(OrderedDict([
            ('block', nn.Sequential(OrderedDict([
                ('exp_1x1', ConvBNAct(in_channels, exp_channels)),
                ('conv_3x3', ConvBNAct(exp_channels, exp_channels, 3, stride, 1, groups=exp_channels)),
                ('red_1x1', ConvBNAct(exp_channels, out_channels))
            ])))
        ]))

    def forward(self, x):
        x = self.conv_1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x
    
class MobileViTV3PAD(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(MobileViTV3PAD, self).__init__()
        # Initialize backbone
        self.backbone = MobileViTv3_XS(num_classes=1000)
        
        if pretrained:
            try:
                weights_path = "./pretrained/MobileVitV3_XS.pt"
                state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
                # Remove unexpected keys
                for key in list(state_dict.keys()):
                    if 'num_batches_tracked' in key:
                        del state_dict[key]
                self.backbone.load_state_dict(state_dict, strict=False)
                print("Loaded pretrained MobileViTv3 weights")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
        
        feature_dim = 640
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Freeze backbone initially
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def unfreeze_layers(self, num_layers=None):
        """Unfreeze layers for fine-tuning"""
        if num_layers is None:
            # Unfreeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = True
                print("Unfroze all backbone layers")
        else:
            # Unfreeze last n layers
            layers = list(self.backbone.named_parameters())
            for name, param in layers[-num_layers:]:
                param.requires_grad = True
                print(f"Unfroze layer: {name}")

    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        
        # Apply attention
        attention = self.attention(features)
        
        # Get classification output
        out = self.classifier(features)
        
        # Apply attention
        return out * attention
 
x = MobileViTV3PAD()