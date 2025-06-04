import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np

class CelebASpoofDataset(Dataset):
    def __init__(self, root_dir, json_path, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images
            json_path (string): Path to the json file with annotations
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load annotations
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
            
        self.image_paths = list(self.annotations.keys())
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        full_img_path = os.path.join(self.root_dir, img_path)
        
        image = Image.open(full_img_path).convert('RGB')
        
        labels = self.annotations[img_path]
        
        spoof_label = torch.tensor(labels[43], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, spoof_label

def get_dataloaders(data_dir, batch_size=32):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CelebASpoofDataset(
        root_dir=data_dir,
        json_path=os.path.join(data_dir, 'metas/intra_test/train_label.json'),
        transform=transform
    )
    
    test_dataset = CelebASpoofDataset(
        root_dir=data_dir,
        json_path=os.path.join(data_dir, 'metas/intra_test/test_label.json'),
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader
