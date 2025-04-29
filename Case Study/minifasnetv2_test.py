import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset.VFPAD_data import VFPADDataset, VFPADTorchDataset

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
    
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    """Training function for the PAD model"""
    best_acer = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'train_acc': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_far': [],
        'val_frr': [],
        'val_acer': []
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate training statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_scores = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                scores = torch.sigmoid(outputs)
                predicted = (scores > 0.5).float()
                
                running_loss += loss.item() * inputs.size(0)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        val_auc = auc(fpr, tpr)
        val_precision = precision_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        val_far, val_frr, val_acer = calculate_far_frr_acer(all_labels, all_preds)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_far'].append(val_far)
        history['val_frr'].append(val_frr)
        history['val_acer'].append(val_acer)
        
        # Print epoch metrics
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
        print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
        print(f'Val FAR: {val_far:.4f}, Val FRR: {val_frr:.4f}, Val ACER: {val_acer:.4f}')
        
        # Step the scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_auc) 
        else:
            scheduler.step()
        
        # Save best model
        if val_acer < best_acer:
            best_acer = val_acer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_acer': val_acer,
            }, './new_models/best_model_minifasnetv2_pad.pth')
            print(f'New best model saved with ACER: {val_acer:.4f}')
    
    return model, history


def plot_training_history(history):
    """Plot training and validation metrics"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(20, 10))
    
    # First row
    plt.subplot(2, 4, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 4, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 4, 3)
    plt.plot(epochs, history['val_auc'], 'g-', label='Validation AUC')
    plt.title('Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.subplot(2, 4, 4)
    plt.plot(epochs, history['val_precision'], 'r-', label='Precision')
    plt.plot(epochs, history['val_recall'], 'g-', label='Recall')
    plt.plot(epochs, history['val_f1'], 'b-', label='F1-Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    
    # Second row - ACER plots
    plt.subplot(2, 4, 6)
    plt.plot(epochs, history['val_far'], 'r-', label='FAR')
    plt.plot(epochs, history['val_frr'], 'g-', label='FRR')
    plt.plot(epochs, history['val_acer'], 'b-', label='ACER')
    plt.title('FAR/FRR/ACER Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./results/minifasnet_training_history.png')
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
    plt.savefig('./results/minifasnetv2_roc_curve.png')
    plt.show()
    
    return roc_auc, fpr, tpr, thresholds

def calculate_far_frr_acer(labels, predictions):
    """Calculate FAR, FRR and ACER metrics"""
    # Convert to numpy arrays if needed
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    # Real/Genuine samples (label = 1)
    real_samples = (labels == 1)
    # Attack/Spoof samples (label = 0)
    attack_samples = (labels == 0)
    
    # False Rejection Rate (FRR): False negative rate for real samples
    frr = np.sum((predictions == 0) & real_samples) / np.sum(real_samples)
    
    # False Acceptance Rate (FAR): False positive rate for attack samples
    far = np.sum((predictions == 1) & attack_samples) / np.sum(attack_samples)
    
    # Average Classification Error Rate (ACER)
    acer = (far + frr) / 2
    
    return far, frr, acer

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
    model = MiniFASNetV2PAD(num_classes=1, pretrained=True)
    model = model.to(device)
    
    # Phase 1: Train only the new layers
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW([
        {'params': model.attention.parameters(), 'lr': 3e-4},
        {'params': model.backbone.classifier.parameters(), 'lr': 3e-4}
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=3,
        verbose=True
    )
    
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
    model.unfreeze_layers(num_layers=10)

    # Create parameter groups with non-overlapping parameters
    backbone_params = []
    attention_params = []
    classifier_params = []

    # Separate parameters into distinct groups
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only include unfrozen parameters
            if 'attention' in name:
                attention_params.append(param)
            elif 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

    # Create optimizer with distinct parameter groups
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': attention_params, 'lr': 3e-4},
        {'params': classifier_params, 'lr': 3e-4}
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=3,
        verbose=True
    )
    
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
        'val_auc': history_phase1['val_auc'] + history_phase2['val_auc'],
        'val_precision': history_phase1['val_precision'] + history_phase2['val_precision'],
        'val_recall': history_phase1['val_recall'] + history_phase2['val_recall'],
        'val_f1': history_phase1['val_f1'] + history_phase2['val_f1'],
        'val_far': history_phase1['val_far'] + history_phase2['val_far'],
        'val_frr': history_phase1['val_frr'] + history_phase2['val_frr'],
        'val_acer': history_phase1['val_acer'] + history_phase2['val_acer']
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