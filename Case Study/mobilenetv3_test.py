import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm
from VFPAD_data import VFPADDataset, VFPADTorchDataset

class MobileNetPAD(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(MobileNetPAD, self).__init__()
        # Load MobileNetV3-Small
        self.backbone = mobilenet_v3_small(pretrained=pretrained)
        
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    '''Training function with validation'''
    best_auc = 0.0
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_auc': [],
        'train_acc': [],  # Added train accuracy
        'val_acc': []     # Added validation accuracy
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
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
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # Print epoch metrics
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
        
        scheduler.step()
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
            }, 'new_models/best_model_mobilenetv3_pad.pth')
            print(f'New best model saved with AUC: {val_auc:.4f}')
    
    return model, history

def plot_training_history(history):
    '''Plot training and validation metrics'''
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
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
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
    plt.savefig('mobilenetv3_training_history.png')
    plt.show()

def plot_roc_curve(model, val_loader, device):
    '''Plot ROC curve for the model using validation data'''
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
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('mobilenetv3_roc_curve.png')
    plt.show()
    
    return roc_auc, fpr, tpr, thresholds


if __name__ == '__main__':
    # Set device and print info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_path = 'VFPAD'
    
    # Load training data with checks
    train_dataset = VFPADDataset(dataset_path, protocol='grandtest', subset='train')
    train_samples = train_dataset.load_data()
    
    if not train_samples:
        raise ValueError('No training samples were loaded! Check dataset path and structure.')
    print(f'Loaded {len(train_samples)} training samples')
    
    # Load validation data with checks
    val_dataset = VFPADDataset(dataset_path, protocol='grandtest', subset='dev')
    val_samples = val_dataset.load_data()
    
    if not val_samples:
        raise ValueError('No validation samples were loaded! Check dataset path and structure.')
    print(f'Loaded {len(val_samples)} validation samples')
    
    # Create datasets with size checks
    train_torch_dataset = VFPADTorchDataset(train_samples, transform=transform)
    val_torch_dataset = VFPADTorchDataset(val_samples, transform=transform)
    
    print(f'Train dataset size: {len(train_torch_dataset)}')
    print(f'Validation dataset size: {len(val_torch_dataset)}')
    
    # Create data loaders with error checking
    if len(train_torch_dataset) > 0:
        train_loader = DataLoader(
            train_torch_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    else:
        raise ValueError('Training dataset is empty!')
    
    if len(val_torch_dataset) > 0:
        val_loader = DataLoader(
            val_torch_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        raise ValueError('Validation dataset is empty!')
    
    # Initialize model with pretrained weights
    model = MobileNetPAD(pretrained=True, num_classes=1)
    model = model.to(device)
    print('Model initialized and moved to device')
    
    # Phase 1: Train only new layers
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([
        {'params': model.attention.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=0.001)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print('Phase 1: Training new layers...')
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
    print('Phase 2: Fine-tuning...')
    model.unfreeze_layers(num_layers=10)  # Unfreeze last 10 layers
    
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
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

    print('\nComputing ROC curve...')
    roc_auc, fpr, tpr, thresholds = plot_roc_curve(model, val_loader, device)

    operating_points = [0.1, 0.05, 0.01]  # FPR targets
    print('\nOperating points:')
    for target_fpr in operating_points:
        idx = np.argmin(np.abs(fpr - target_fpr))
        print(f'At FPR = {fpr[idx]:.4f}:')
        print(f'  TPR = {tpr[idx]:.4f}')
        print(f'  Threshold = {thresholds[idx]:.4f}')
    

    plot_training_history(history)