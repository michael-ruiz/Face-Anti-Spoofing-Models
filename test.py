import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Dataset class
class CelebASpoofDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset folder (e.g., 'CelebA_Spoof/Data/test').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        """Creates a list of (image_path, label) tuples from the dataset structure."""
        samples = []
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                for class_name in ["live", "spoof"]:
                    class_path = os.path.join(folder_path, class_name)
                    if os.path.exists(class_path):
                        for file in os.listdir(class_path):
                            if file.endswith(".png"):
                                img_path = os.path.join(class_path, file)
                                txt_path = os.path.join(class_path, file.replace(".png", ".txt"))
                                label = self._get_label(txt_path)
                                samples.append((img_path, label))
        return samples

    def _get_label(self, txt_path):
        """Extracts the live/spoof label from the text file."""
        try:
            with open(txt_path, "r") as f:
                values = f.readline().strip().split()
                return int(values[0])  # Assuming the first value is the label (0: spoof, 1: live)
        except:
            return 0  # Default to spoof if there's an error

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Define MobileNet model
class MobileNetFAS(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetFAS, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)  # Modify for binary classification

    def forward(self, x):
        return self.model(x)

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10, device="cuda"):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Evaluation function
def evaluate(model, val_loader, criterion, device="cuda"):
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    return val_acc, val_loss / len(val_loader)

# Main script execution
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define dataset transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = CelebASpoofDataset(root_dir="CelebA_Spoof/Data/test", transform=transform)
    
    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)  # num_workers=0 for Windows safety
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize model, loss function, and optimizer
    model = MobileNetFAS(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, epochs=10, device=device)

    # Save the trained model
    torch.save(model.state_dict(), "mobilenet_fas.pth")
    print("Model saved!")
