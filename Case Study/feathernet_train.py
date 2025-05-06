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
from models.feathernetb import FeatherNetPAD

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
            }, './new_models/best_model_feathernet_pad.pth')
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
    plt.savefig('./results/feathernet_training_history.png')
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
    plt.savefig('./results/feathernet_roc_curve.png')
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
    model = FeatherNetPAD(num_classes=1, width_mult=1.0, scale=1.0, pretrained=True)
    model = model.to(device)
    
    # Phase 1: Train only the new layers
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW([
        {'params': model.attention.parameters(), 'lr': 3e-4},
        {'params': model.classifier.parameters(), 'lr': 3e-4}
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
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
    model.unfreeze_layers(num_layers=10)  # Unfreeze last 10 layers
    
    optimizer = optim.AdamW([
        {'params': model.feathernet.parameters(), 'lr': 1e-4},
        {'params': model.attention.parameters(), 'lr': 3e-4},
        {'params': model.classifier.parameters(), 'lr': 3e-4}
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
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