import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from dataset.VFPAD_data import VFPADDataset, VFPADTorchDataset
from feathernet_train import FeatherNetPAD
from minifasnetv2_train import MiniFASNetV2PAD
from models.mobilenetv3 import MobileNetV3PAD

def load_model(model_class, model_path, device):
    """Load a model from checkpoint"""
    model = model_class(num_classes=1, pretrained=False)
    
    # Add safe globals for numpy scalar types
    torch.serialization.add_safe_globals([
        np.int64, np.float32, np.float64,
        np._core.multiarray.scalar
    ])
    
    # Load checkpoint with weights_only=False since we trust our own checkpoints
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def test_inference_speed(model, test_loader, device):
    """Test model inference speed with entire test dataset"""
    model.eval()
    # Get all test images
    test_images = []
    for inputs, _ in tqdm(test_loader, desc='Loading test images'):
        test_images.append(inputs)
    
    num_iterations = len(test_images)
    print(f"Testing inference speed on {num_iterations} images")
    
    # Warm-up with a few iterations
    # for _ in range(min(10, num_iterations)):
    #     _ = model(test_images[0].to(device))
    
    # Measure inference time
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for image in tqdm(test_images, desc='Running inference'):
            _ = model(image.to(device))
            torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    return fps, avg_time * 1000  # Return FPS and time in milliseconds

def evaluate_random_samples(model, test_loader, device, num_samples=10):
    """Evaluate model on random test samples"""
    # Get random indices
    dataset_size = len(test_loader.dataset)
    random_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    results = []
    model.eval()
    
    for idx in random_indices:
        # Get sample
        image, label = test_loader.dataset[idx]
        image = image.unsqueeze(0).to(device)
        
        # Measure inference time
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = model(image)
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate prediction
        score = torch.sigmoid(output).cpu().numpy()[0, 0]
        pred = 1 if score > 0.5 else 0
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Convert label to int if it's a tensor
        if torch.is_tensor(label):
            label = label.item()
        
        results.append({
            'true_label': label,
            'predicted': pred,
            'confidence': score,
            'inference_time': inference_time
        })
    
    return results

def print_model_info(model, model_name):
    """Print model parameters and architecture information"""
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{model_name} Architecture Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024*1024):.2f} MB")  # Assuming float32 (4 bytes)
    
    # Print detailed architecture
    print("\nModel Architecture:")
    print(model)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    dataset_path = "VFPAD"
    test_dataset = VFPADDataset(dataset_path, protocol='grandtest', subset='eval')
    test_samples = test_dataset.load_data()
    test_torch_dataset = VFPADTorchDataset(test_samples, transform=transform)
    
    test_loader = DataLoader(
        test_torch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model configurations
    models_config = {
        'FeatherNet': {
            'class': FeatherNetPAD,
            'path': './new_models/best_model_feathernet_pad.pth'
        },
        'MiniFASNetV2': {
            'class': MiniFASNetV2PAD,
            'path': './new_models/best_model_minifasnetv2_pad.pth'
        },
        'MobileNetV3': {
            'class': MobileNetV3PAD,
            'path': './new_models/best_model_mobilenetv3_pad.pth'
        }
    }
    
    # Test each model
    results = {}
    for model_name, config in models_config.items():
        print(f"\nTesting {model_name}...")
        try:
            # Load model
            model = load_model(config['class'], config['path'], device)

            # Print model information
            print_model_info(model, model_name)
            
            # Test inference speed
            fps, avg_time = test_inference_speed(model, test_loader, device)
            print(f"\n{model_name} Inference Speed on Full Test Set:")
            print(f"  Number of Images: {len(test_loader.dataset)}")
            print(f"  Processing Speed: {fps:.2f} frames/second")
            print(f"  Average Time per Frame: {avg_time:.2f} milliseconds")
            
            # Evaluate random samples
            sample_results = evaluate_random_samples(model, test_loader, device)
            
            results[model_name] = {
                'fps': fps,
                'avg_time': avg_time,
                'sample_results': sample_results
            }
            
            # Print sample results
            print(f"\n{model_name} Random Sample Results:")
            for i, result in enumerate(sample_results):
                print(f"\nSample {i+1}:")
                print(f"  True Label: {'Real' if result['true_label'] == 1 else 'Spoof'}")
                print(f"  Predicted: {'Real' if result['predicted'] == 1 else 'Spoof'}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Inference Time: {result['inference_time']:.2f} ms")
                
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    fps_values = [results[model]['fps'] for model in model_names]
    
    plt.bar(model_names, fps_values)
    plt.title('Model Inference Speed Comparison')
    plt.ylabel('Frames per Second (FPS)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_speed_comparison.png')
    plt.show()

if __name__ == '__main__':
    main()