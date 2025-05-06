import torch
import torch.nn as nn

class MiniFASNetV2(nn.Module):
    def __init__(self, num_classes=1, embedding_size=128, channels=32):
        super(MiniFASNetV2, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Feature extraction blocks
        self.layer1 = self._make_layer(channels, channels, num_blocks=4)
        self.layer2 = self._make_layer(channels, channels*2, num_blocks=6, stride=2)
        self.layer3 = self._make_layer(channels*2, channels*4, num_blocks=8, stride=2)
        self.layer4 = self._make_layer(channels*4, channels*8, num_blocks=6, stride=2)
        
        # Global average pooling and FC
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels*8, embedding_size)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(embedding_size, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_blocks-1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 
                         kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = self.fc(x)
        x = self.dropout(feat)
        x = self.classifier(x)
        
        return x

class MiniFASNetV2PAD(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(MiniFASNetV2PAD, self).__init__()
        # Initialize backbone
        self.backbone = MiniFASNetV2(num_classes=128)  # Match embedding size
        
        if pretrained:
            try:
                # Load pretrained weights
                weights_path = "./pretrained/MiniFASNetV2.pth"
                pretrained_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
                model_dict = self.backbone.state_dict()
                
                # Filter out unnecessary weights
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if k in model_dict and 'classifier' not in k}
                model_dict.update(pretrained_dict)
                self.backbone.load_state_dict(model_dict, strict=False)
                print("Loaded pretrained MiniFASNetV2 weights")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                print("Initializing with random weights")
        
        # Initialize new layers properly
        self._initialize_new_layers()
        
        # Freeze backbone layers initially
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace classifier for binary PAD
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _initialize_new_layers(self):
        """Initialize the new layers with proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def unfreeze_layers(self, num_layers=None):
        """Unfreeze layers for fine-tuning"""
        if num_layers is None:
            # Unfreeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Unfroze all backbone layers")
        else:
            # Get all backbone parameters
            layers = list(self.backbone.named_parameters())
            total_layers = len(layers)
            
            # Calculate start index to unfreeze last n layers
            start_idx = max(0, total_layers - num_layers)
            
            # Unfreeze specified number of layers from the end
            for name, param in layers[start_idx:]:
                param.requires_grad = True
                print(f"Unfroze layer: {name}")
    
    def forward(self, x):
        # Extract features through backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        features = self.backbone.fc(x.view(x.size(0), -1))
        
        # Calculate attention weights
        attention = torch.sigmoid(self.attention(features))
        
        # Get classification output
        out = self.backbone.classifier(features)
        
        # Apply attention weights
        return out * attention
    
    def freeze_backbone(self):
        """Freeze all backbone layers"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Froze all backbone layers")
    
    def get_trainable_params(self):
        """Get list of trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]
