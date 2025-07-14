import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
from dataset.VFPAD_data import VFPADDataset, VFPADTorchDataset
import time

# Import all model architectures
from models.feathernetb import FeatherNetPAD
from models.mobilenetv3 import MobileNetV3PAD
from models.mobilenetv4 import MobileNetV4PAD
from models.mobilevitv3 import MobileViTV3PAD
from models.minifasnetv2 import MiniFASNetV2PAD
from models.efficientnetv2b0 import EfficientNetV2PAD

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_labels = []
    all_scores = []
    
    start_time = time.time()
    total_images = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            outputs = model(inputs)
            scores = torch.sigmoid(outputs)
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            total_images += inputs.size(0)
    
    elapsed_time = time.time() - start_time
    fps = total_images / elapsed_time if elapsed_time > 0 else 0

    # Debug info
    print(f"Total images: {total_images}, Elapsed time: {elapsed_time:.4f} sec, FPS: {fps:.2f}")

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    predictions = (all_scores > 0.5).astype(int)
    accuracy = np.mean(predictions == all_labels)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    
    # Calculate APCER, BPCER and ACER
    attack_samples = (all_labels == 1)
    bona_fide_samples = (all_labels == 0)
    
    apcer = np.sum((predictions == 0) & attack_samples) / np.sum(attack_samples)
    bpcer = np.sum((predictions == 1) & bona_fide_samples) / np.sum(bona_fide_samples)
    acer = (apcer + bpcer) / 2
    
    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'acer': acer,
        'apcer': apcer,
        'bpcer': bpcer,
        'auc': roc_auc,
        'fps': fps
    } 

if __name__ == "__main__":
    # Intel® Core™ i9-14900KF

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load evaluation dataset
    dataset_path = "VFPAD"
    eval_dataset = VFPADDataset(dataset_path, protocol='grandtest', subset='eval')
    eval_samples = eval_dataset.load_data()
    eval_torch_dataset = VFPADTorchDataset(eval_samples, transform=transform)
    
    eval_loader = DataLoader(
        eval_torch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Models to evaluate
    models = {
        'FeatherNetB': {
            'class': FeatherNetPAD,
            'path': 'new_models/best_model_feathernet_pad.pth'
        },
        'MobileNetV3': {
            'class': MobileNetV3PAD,
            'path': 'new_models/best_model_mobilenetv3_pad.pth'
        },
        'MobileNetV4': {
            'class': MobileNetV4PAD,
            'path': 'new_models/best_model_mobilenetv4_pad.pth'
        },
        'MobileViTV3': {
            'class': MobileViTV3PAD,
            'path': 'new_models/best_model_mobilevitv3_pad.pth'
        },
        'MiniFASNetV2': {
            'class': MiniFASNetV2PAD,
            'path': 'new_models/best_model_minifasnetv2_pad.pth'
        },
        'EfficientNetV2': {
            'class': EfficientNetV2PAD,
            'path': 'new_models/best_model_efficientnetv2_pad.pth'
        }
    }
    
    results = {}
    
    # Evaluate each model
    overall_start = time.time()
    for name, model_info in models.items():
        print(f"\nEvaluating {name}...")
        
        # Initialize model
        model = model_info['class'](num_classes=1, pretrained=False)
        model.load_state_dict(torch.load(model_info['path'], map_location=device)['model_state_dict'])
        model = model.to(device)
        
        # Evaluate
        metrics = evaluate_model(model, eval_loader, device)
        results[name] = metrics
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 120)
    print(f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ACER':>10} {'APCER':>10} {'BPCER':>10} {'AUC':>10} {'FPS':>10}")
    print("-" * 120)
    
    for name, metrics in results.items():
        print(f"{name:<15} "
              f"{metrics['accuracy']:10.4f} "
              f"{metrics['precision']:10.4f} "
              f"{metrics['recall']:10.4f} "
              f"{metrics['f1']:10.4f} "
              f"{metrics['acer']:10.4f} "
              f"{metrics['apcer']:10.4f} "
              f"{metrics['bpcer']:10.4f} "
              f"{metrics['auc']:10.4f} "
              f"{metrics['fps']:10.2f}")