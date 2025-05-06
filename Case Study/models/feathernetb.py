import torch
import torch.nn as nn

# FeatherNetB for PAD
class FeatherNetB(nn.Module):
    def __init__(self, num_classes=1, input_size=224, width_mult=1.0, scale=1.0):
        super(FeatherNetB, self).__init__()
        self.input_size = input_size
        
        # First conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(32 * width_mult), 3, 2, 1, bias=False),
            nn.BatchNorm2d(int(32 * width_mult)),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction layers
        self.features = self._make_layers(width_mult, scale)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(int(1024 * width_mult), num_classes)
        )
        
        self._initialize_weights()
    
    def _make_layers(self, width_mult, scale):
        layers = []
        
        # FeatherNetB architecture
        # Block 1
        layers.append(self._make_block(int(32 * width_mult), int(64 * width_mult), 2, 2, scale))
        
        # Block 2
        layers.append(self._make_block(int(64 * width_mult), int(128 * width_mult), 2, 2, scale))
        
        # Block 3
        layers.append(self._make_block(int(128 * width_mult), int(256 * width_mult), 2, 2, scale))
        
        # Block 4
        layers.append(self._make_block(int(256 * width_mult), int(512 * width_mult), 2, 1, scale))
        
        # Final Conv
        layers.append(nn.Sequential(
            nn.Conv2d(int(512 * width_mult), int(1024 * width_mult), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(1024 * width_mult)),
            nn.ReLU(inplace=True)
        ))
        
        # Global Pool
        layers.append(nn.AdaptiveAvgPool2d(1))
        
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels, out_channels, stride, repeat, scale):
        blocks = []
        
        # Depthwise separable convolution
        blocks.append(nn.Sequential(
            # Depthwise conv
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise conv
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Additional blocks
        for i in range(1, repeat):
            blocks.append(nn.Sequential(
                # Depthwise conv
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                # Pointwise conv
                nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        return nn.Sequential(*blocks)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Feathernet with attention for PAD
class FeatherNetPAD(nn.Module):
    def __init__(self, num_classes=1, width_mult=1.0, scale=1.0, pretrained=True):
        super(FeatherNetPAD, self).__init__()
        # Initialize backbone
        self.feathernet = FeatherNetB(num_classes=1000, width_mult=width_mult, scale=scale)
        
        if pretrained:
            try:
                weights_path = "./pretrained/feathernet_best.pth.tar"
                pretrained_dict = torch.load(weights_path, map_location='cpu')
                model_dict = self.feathernet.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if k in model_dict and 'classifier' not in k}
                model_dict.update(pretrained_dict)
                self.feathernet.load_state_dict(model_dict)
                print("Loaded pretrained FeatherNet weights")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
        
        # Freeze backbone initially
        for param in self.feathernet.parameters():
            param.requires_grad = False
        
        # Feature dimension
        feature_dim = int(1024 * width_mult)
        
        # Modified attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Modified classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def unfreeze_layers(self, num_layers=None):
        """Unfreeze layers for fine-tuning"""
        if num_layers is None:
            # Unfreeze all backbone layers
            for param in self.feathernet.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last n layers
            layers = list(self.feathernet.named_parameters())
            for name, param in reversed(layers[:num_layers]):
                param.requires_grad = True
                print(f"Unfroze layer: {name}")

    def forward(self, x):
        # Get features
        x = self.feathernet.conv1(x)
        features = self.feathernet.features[:-1](x)  # Skip global pooling
        
        # Global average pooling
        features = torch.mean(features, dim=(2, 3))  # (batch_size, channels)
        
        # Get attention weights
        attention = torch.sigmoid(self.attention(features))
        
        # Get classification output
        out = self.classifier(features)
        
        # Apply attention
        return out * attention
