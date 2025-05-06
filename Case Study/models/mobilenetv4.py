import torch
import torch.nn as nn
import timm

class MobileNetV4PAD(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(MobileNetV4PAD, self).__init__()
        # Load pretrained MobileNetV4 without classifier
        self.backbone = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k', 
            pretrained=pretrained,
            features_only=True,  # Get intermediate features
            out_indices=(4,)     # Only get the last feature map
        )
        
        # Get feature dimension from the last layer
        dummy_input = torch.randn(1, 3, 224, 224)
        features = self.backbone(dummy_input)
        feature_dim = features[0].shape[1]
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        if pretrained:
            # Freeze backbone initially
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_layers(self, num_layers=None):
        """Unfreeze layers for fine-tuning"""
        if num_layers is None:
            # Unfreeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last n layers
            layers = list(self.backbone.named_parameters())
            for name, param in reversed(layers[:num_layers]):
                param.requires_grad = True
                print(f"Unfroze layer: {name}")

    def forward(self, x):
        # Get features from backbone (now returns a tuple with one tensor)
        features = self.backbone(x)
        
        # Use the single feature map
        x = features[0]
        
        # Global average pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Apply attention
        attention = self.attention(x)
        
        # Get classification output
        out = self.classifier(x)
        
        # Apply attention
        return out * attention
