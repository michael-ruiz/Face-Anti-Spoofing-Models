import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from mobilenet_test_2 import VFPADDataset, MobileNetV3PAD

def visualize_prediction(image, prediction, label, prob):
    """Visualize the image and prediction"""
    plt.figure(figsize=(8, 8))
    
    # Convert image from tensor to numpy for visualization
    if torch.is_tensor(image):
        image = image.cpu().numpy().transpose(1, 2, 0)
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    
    plt.imshow(image)
    plt.axis('off')
    
    # Add prediction text
    color = 'green' if prediction == label else 'red'
    plt.title(f'Prediction: {"Attack" if prediction == 1 else "Bona Fide"}\n'
             f'True Label: {"Attack" if label == 1 else "Bona Fide"}\n'
             f'Confidence: {prob:.2f}', 
             color=color)
    
    plt.show()

def test_random_sample():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset_path = "VFPAD"
    test_dataset = VFPADDataset(dataset_path, protocol='grandtest', subset='eval')
    test_samples = test_dataset.load_data()
    
    # Select a random sample
    sample = random.choice(test_samples)
    
    # Load model
    model = MobileNetV3PAD(num_classes=1, pretrained=False)  # Set pretrained=False since we're loading weights
    model.load_state_dict(torch.load('best_model_mobilenetv3_pad.pth', map_location=device)['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Process frame
    frames = sample['frames']
    if not frames:
        print("No frames found in sample")
        return
    
    # Take first frame
    frame = frames[0]
    if frame.ndim == 2:  # Grayscale
        frame = np.stack([frame] * 3, axis=2)
    
    # Apply transforms
    frame_tensor = transform(frame).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(frame_tensor)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > 0.5 else 0
    
    # Get true label
    true_label = 1 if sample['is_attack'] else 0
    
    # Visualize result
    visualize_prediction(transform(frame), prediction, true_label, prob)
    
    # Print additional information
    print("\nSample Information:")
    print(f"Filename: {sample['filename']}")
    if 'metadata' in sample:
        print(f"Type: {sample['metadata'].get('type_description', 'Unknown')}")
        print(f"Subcategory: {sample['metadata'].get('subcategory_description', 'Unknown')}")

if __name__ == "__main__":
    test_random_sample()