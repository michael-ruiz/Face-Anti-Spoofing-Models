import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from VFPAD_data import VFPADDataset, VFPADTorchDataset


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
        # Initialize the FeatherNetB model
        self.feathernet = FeatherNetB(num_classes=1000, width_mult=width_mult, scale=scale)  # Initially 1000 classes for ImageNet
        
        if pretrained:
            # Load pretrained weights
            weights_path = "./pretrained/feathernet_best.pth.tar"            
            # Load pretrained weights
            pretrained_dict = torch.load(weights_path, map_location='cpu')
            model_dict = self.feathernet.state_dict()
            
            # Filter out classifier weights
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and 'classifier' not in k}
            model_dict.update(pretrained_dict)
            self.feathernet.load_state_dict(model_dict)
            print("Loaded pretrained FeatherNet weights")
        
        # Freeze backbone layers
        for param in self.feathernet.parameters():
            param.requires_grad = False
            
        # Replace classifier for PAD task
        feature_dim = int(1024 * width_mult * 7 * 7)
        self.attention = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_dim, int(512 * width_mult)),
            nn.ReLU(inplace=True),
            nn.Linear(int(512 * width_mult), num_classes)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(int(1024 * width_mult), num_classes)
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
        # Extract features from FeatherNetB
        x = self.feathernet.conv1(x)
        x = self.feathernet.features[:-1](x)  # Exclude global pooling
        
        # Store the original shape for attention
        batch_size, channels, height, width = x.shape
        
        # Calculate attention weights
        # Flatten the spatial dimensions for attention
        flattened = x.view(batch_size, -1)
        attention_weights = torch.sigmoid(self.attention(flattened))
        
        # Apply global pooling and continue with classification
        x = self.feathernet.features[-1](x)  # Global pooling
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        # Weight the output with attention
        x = x * attention_weights
        
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    """Training function for the PAD model"""
    best_auc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        
        # Store training metrics
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                scores = torch.sigmoid(outputs)
                
                running_loss += loss.item() * inputs.size(0)
                predicted = (scores > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        val_auc = auc(fpr, tpr)
        
        # Store validation metrics
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # Print epoch metrics
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
        
        # Step the scheduler
        scheduler.step()
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, './new_models/best_model_feathernet_pad.pth')
            print(f'New best model saved with AUC: {val_auc:.4f}')
    
    return model, history



def plot_training_history(history):
    """Plot training and validation metrics"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot AUC
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_auc'], 'g-', label='Validation AUC')
    plt.title('Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('feathernet_training_history.png')
    plt.show()

def plot_roc_curve(model, val_loader, device):
    """Plot ROC curve for the model using validation data"""
    model.eval()
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Computing ROC'):
            inputs = inputs.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            outputs = model(inputs)
            scores = torch.sigmoid(outputs)
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('feathernet_roc_curve.png')
    plt.show()
    
    return roc_auc, fpr, tpr, thresholds


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize dataset
    dataset_path = "VFPAD"
    
    # Load training data
    train_dataset = VFPADDataset(dataset_path, protocol='grandtest', subset='train')
    train_samples = train_dataset.load_data()
    train_torch_dataset = VFPADTorchDataset(train_samples, transform=transform)
    
    # Load validation data
    val_dataset = VFPADDataset(dataset_path, protocol='grandtest', subset='dev')
    val_samples = val_dataset.load_data()
    val_torch_dataset = VFPADTorchDataset(val_samples, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_torch_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_torch_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model with pretrained weights
    model = FeatherNetPAD(num_classes=1, width_mult=1.0, scale=1.0, pretrained=True)
    model = model.to(device)
    
    # Phase 1: Train only the new layers
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([
        {'params': model.attention.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=0.001)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print("Phase 1: Training new layers...")
    model, history_phase1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        device=device
    )
    
    # Phase 2: Fine-tune the model
    print("Phase 2: Fine-tuning...")
    model.unfreeze_layers(num_layers=10)  # Unfreeze last 10 layers
    
    optimizer = optim.Adam([
        {'params': model.feathernet.parameters(), 'lr': 1e-5},
        {'params': model.attention.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    model, history_phase2 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=20,
        device=device
    )
    
    history = {
        'train_loss': history_phase1['train_loss'] + history_phase2['train_loss'],
        'val_loss': history_phase1['val_loss'] + history_phase2['val_loss'],
        'train_acc': history_phase1['train_acc'] + history_phase2['train_acc'],
        'val_acc': history_phase1['val_acc'] + history_phase2['val_acc'],        
        'val_auc': history_phase1['val_auc'] + history_phase2['val_auc']
    }

    print("\nComputing ROC curve...")
    roc_auc, fpr, tpr, thresholds = plot_roc_curve(model, val_loader, device)

    operating_points = [0.1, 0.05, 0.01]  # FPR targets
    print("\nOperating points:")
    for target_fpr in operating_points:
        idx = np.argmin(np.abs(fpr - target_fpr))
        print(f"At FPR = {fpr[idx]:.4f}:")
        print(f"  TPR = {tpr[idx]:.4f}")
        print(f"  Threshold = {thresholds[idx]:.4f}")
    
    plot_training_history(history)