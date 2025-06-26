import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from CelebA_Spoof import get_dataloaders
from mobilenetv4 import MobileNetV4PAD
from efficientnetv2b0 import EfficientNetV2PAD
import time
import numpy as np

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                total += labels.size(0)
        
        val_loss = running_loss / total
        val_acc = running_corrects.double() / total
        
        print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {best_acc:.4f}')
        

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get dataloaders
    dataset_path = "CelebA_Spoof"
    train_loader, test_loader = get_dataloaders(dataset_path, batch_size=32)
    
    # Training configurations
    models = {
        'MobileNetV4': MobileNetV4PAD(pretrained=False, num_classes=2),
        'EfficientNetV2': EfficientNetV2PAD(pretrained=False, num_classes=2)
    }
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}")
        print("=" * 50)
        
        model = model.to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
        
        # Train the model
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=15,
            device=device
        )
        
        # Save final model
        torch.save(model.state_dict(), f'best_{model_name.lower()}.pth')

if __name__ == "__main__":
    main()