import torch
import torch.nn as nn
import timm

class EfficientNetV2PAD(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetV2PAD, self).__init__()
        # Initialize backbone with pretrained weights
        self.backbone = timm.create_model('tf_efficientnetv2_b0', pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        feature_dim = 1280
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        
        # Add global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Freeze backbone initially
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
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
        
        # Apply global pooling and flatten
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        # Apply attention
        attention = self.attention(features)
        
        # Get classification output
        out = self.classifier(features)
        
        # Apply attention
        return out * attention
