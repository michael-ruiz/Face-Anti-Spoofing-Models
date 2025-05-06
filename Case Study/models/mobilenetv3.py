import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class MobileNetV3PAD(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(MobileNetV3PAD, self).__init__()
        # Load MobileNetV3-Small
        self.backbone = mobilenet_v3_small(pretrained=pretrained, weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
        
        # Get the number of features from the last layer
        in_features = self.backbone.classifier[0].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Add classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Freeze backbone initially if using pretrained weights
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def unfreeze_layers(self, num_layers=None):
        '''Unfreeze layers for fine-tuning'''
        if num_layers is None:
            # Unfreeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = True
            print('Unfroze all backbone layers')
        else:
            # Get all feature layers
            feature_layers = list(self.backbone.features.named_parameters())
            total_layers = len(feature_layers)
            
            # Calculate how many layers to unfreeze from the end
            start_idx = max(0, total_layers - num_layers)
            
            # Unfreeze the last num_layers
            for name, param in feature_layers[start_idx:]:
                param.requires_grad = True
                print(f'Unfroze layer: {name}')
    
    def forward(self, x):
        # Extract features through backbone
        features = self.backbone.features(x)
        # Apply adaptive average pooling
        features = self.backbone.avgpool(features)
        # Flatten
        features = torch.flatten(features, 1)
        
        # Calculate attention weights
        attention_weights = self.attention(features)
        
        # Apply attention and get classification
        out = self.classifier(features)
        out = out * attention_weights
        
        return out
