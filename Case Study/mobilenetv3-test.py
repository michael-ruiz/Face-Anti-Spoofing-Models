import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import h5py

class VFPADDataset:
    '''
    A class to load and process the VFPAD (in-Vehicle Face Presentation Attack Detection) dataset.
    '''
    
    def __init__(self, dataset_path, protocol='grandtest', subset='train'):
        '''
        Initialize the VFPAD dataset loader.
        
        Args:
            dataset_path (str): Path to the VFPAD dataset
            protocol (str): Protocol to use, default is 'grandtest'
            subset (str): Subset to load ('train', 'dev', or 'eval')
        '''
        self.dataset_path = dataset_path
        self.protocol = protocol
        self.subset = subset
        
        # Print the absolute path to help with debugging
        abs_path = os.path.abspath(dataset_path)
        print(f'Loading VFPAD dataset from: {abs_path}')
        
        # Check if protocol directory exists
        protocol_dir = os.path.join(dataset_path, 'protocol', protocol)
        if not os.path.exists(protocol_dir):
            print(f'Warning: Protocol directory not found: {protocol_dir}')
            # Try alternate paths
            alt_protocol_dir = os.path.join(dataset_path, 'protocols', protocol)
            if os.path.exists(alt_protocol_dir):
                protocol_dir = alt_protocol_dir
                print(f'Found alternate protocol directory: {protocol_dir}')
        
        # Load protocol files
        self.bona_fide_files = []
        self.attack_files = []
        
        self._load_protocol_files()
    
    def _load_protocol_files(self):
        '''Load the protocol files for the specified subset.'''
        # Try multiple possible paths for protocol files
        possible_protocol_paths = [
            os.path.join(self.dataset_path, 'protocol', self.protocol, self.subset),
            os.path.join(self.dataset_path, 'protocols', self.protocol, self.subset),
            os.path.join(self.dataset_path, 'protocol', 'grandtest', self.subset),
            os.path.join(self.dataset_path, 'protocols', 'grandtest', self.subset)
        ]
        
        protocol_path = None
        for path in possible_protocol_paths:
            if os.path.exists(path):
                protocol_path = path
                print(f'Using protocol path: {path}')
                break
        
        if not protocol_path:
            print(f'Error: Could not find protocol path for {self.subset} subset')
            print(f'Tried the following paths:')
            for path in possible_protocol_paths:
                print(f'  - {path}')
            return
        
        # Load bona fide files
        bf_protocol_file = os.path.join(protocol_path, 'for_real.lst')
        if os.path.exists(bf_protocol_file):
            print(f'Loading bona fide protocol file: {bf_protocol_file}')
            with open(bf_protocol_file, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                print(f'Found {len(lines)} lines in bona fide protocol file')
                
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename, client_id = parts[0], parts[1]
                        self.bona_fide_files.append({
                            'filename': filename,
                            'client_id': client_id,
                            'is_attack': False
                        })
        else:
            print(f'Bona fide protocol file not found: {bf_protocol_file}')
        
        # Load attack files
        attack_protocol_file = os.path.join(protocol_path, 'for_attack.lst')
        if os.path.exists(attack_protocol_file):
            print(f'Loading attack protocol file: {attack_protocol_file}')
            with open(attack_protocol_file, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                print(f'Found {len(lines)} lines in attack protocol file')
                
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        filename, client_id, attack_type = parts[0], parts[1], parts[2]
                        self.attack_files.append({
                            'filename': filename,
                            'client_id': client_id,
                            'attack_type': attack_type,
                            'is_attack': True
                        })
        else:
            print(f'Attack protocol file not found: {attack_protocol_file}')
        
        print(f'Loaded {len(self.bona_fide_files)} bona fide files and {len(self.attack_files)} attack files')
    
    def parse_filename(self, filename):
        '''
        Parse the filename to extract metadata.
        
        The filename format is:
        <presentation-type>_<session-id>_<angle-id>_<illumination-id>_<client-id>_<presenter-id>_<type-id>_<sub-category-id>_<pai-id>_<trial-id>.hdf5
        '''
        parts = filename.split('_')
        if len(parts) < 10:
            return None
        
        metadata = {
            'presentation_type': parts[0],  # 'bf' or 'pa'
            'session_id': parts[1],         # '01', '02', '03', or '04'
            'angle_id': parts[2],           # '1' or '2'
            'illumination_id': parts[3],    # '1' or '2'
            'client_id': parts[4],          # identity of subject or PAI
            'presenter_id': parts[5],       # '0000' for bf, '0001' for pa
            'type_id': parts[6],            # '00', '01', '02', '03', or '04'
            'sub_category_id': parts[7],    # depends on type_id
            'pai_id': parts[8],             # '000' for bf, unique number for pa
            'trial_id': parts[9]            # arbitrary numeric string
        }
        
        # Add human-readable descriptions
        type_descriptions = {
            '00': 'bona-fide',
            '01': 'print',
            '02': 'replay-attack',
            '03': '3D silicone mask',
            '04': '3D rigid mask'
        }
        
        subcategory_descriptions = {
            '00': {
                '00': 'Natural (no glasses or hat)',
                '01': 'Medical glasses',
                '02': 'Clear glasses',
                '03': 'Sunglasses',
                '04': 'Hat (no glasses)',
                '05': 'Hat + clear glasses',
                '06': 'Hat + sunglasses'
            },
            '01': {
                '01': 'Matte on Laser printer',
                '02': 'Glossy on Laser printer',
                '03': 'Matte on Inkjet printer',
                '04': 'Glossy on Inkjet printer'
            },
            '02': {
                '00': 'Replay attack'
            },
            '03': {
                '00': 'Generic flexible mask (G-Flex-3D-Mask)',
                '01': 'Custom flexible mask (C-Flex-3D-Mask)'
            },
            '04': {
                '00': 'Custom rigid mask 1',
                '01': 'Custom rigid mask 2',
                '02': 'Custom rigid mask 3',
                '03': 'Custom rigid mask 4'
            }
        }
        
        metadata['type_description'] = type_descriptions.get(metadata['type_id'], 'Unknown')
        
        if metadata['type_id'] in subcategory_descriptions and metadata['sub_category_id'] in subcategory_descriptions[metadata['type_id']]:
            metadata['subcategory_description'] = subcategory_descriptions[metadata['type_id']][metadata['sub_category_id']]
        else:
            metadata['subcategory_description'] = 'Unknown'
        
        return metadata
    
    def load_sample(self, file_info):
        '''
        Load a single sample from the dataset.
        
        Args:
            file_info (dict): Information about the file to load
            
        Returns:
            dict: The loaded sample including frames and metadata
        '''
        # Try multiple possible paths for data
        possible_data_paths = [
            os.path.join(self.dataset_path, 'data'),
            os.path.join(self.dataset_path, 'preprocessed')
        ]
        
        file_path = None
        for data_path in possible_data_paths:
            # Check both with and without file extension
            test_path = os.path.join(data_path, f'{file_info['filename']}.hdf5')
            if os.path.exists(test_path):
                file_path = test_path
                break
                
            # Also try looking in bf/pa subdirectories
            for prefix in ['bf', 'pa']:
                test_path = os.path.join(data_path, prefix, f'{file_info['filename']}.hdf5')
                if os.path.exists(test_path):
                    file_path = test_path
                    break
                
                # Try looking in client ID subdirectories
                # For example: bf/0006/bf_03_1_1_0006_0000_00_02_000_10667511.hdf5
                client_id_match = file_info['filename'].split('_')[4]
                test_path = os.path.join(data_path, prefix, client_id_match, f'{file_info['filename']}.hdf5')
                if os.path.exists(test_path):
                    file_path = test_path
                    break
        
        if not file_path:
            print(f'File not found for: {file_info['filename']}')
            # Print the paths that were tried
            for data_path in possible_data_paths:
                print(f'  Tried: {os.path.join(data_path, file_info['filename'])}.hdf5')
                for prefix in ['bf', 'pa']:
                    print(f'  Tried: {os.path.join(data_path, prefix, file_info['filename'])}.hdf5')
                    client_id_match = file_info['filename'].split('_')[4]
                    print(f'  Tried: {os.path.join(data_path, prefix, client_id_match, file_info['filename'])}.hdf5')
            return None
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Print the contents of the HDF5 file for debugging
                print(f'HDF5 file structure for {os.path.basename(file_path)}:')
                self._print_hdf5_structure(f)
                
                metadata = self.parse_filename(os.path.basename(file_path).split('.')[0])
                
                # Try to load frames based on file structure
                frames = self._extract_frames_from_hdf5(f, metadata)
                
                sample = {
                    'filename': file_info['filename'],
                    'frames': frames,
                    'metadata': metadata,
                    'is_attack': file_info.get('is_attack', False)
                }
                
                if 'attack_type' in file_info:
                    sample['attack_type'] = file_info['attack_type']
                
                return sample
        except Exception as e:
            print(f'Error loading file {file_path}: {e}')
            return None
    
    def _print_hdf5_structure(self, hdf5_file, indent=0):
        '''Print the structure of an HDF5 file for debugging.'''
        for key in hdf5_file.keys():
            item = hdf5_file[key]
            print('  ' * indent + f'/{key} ({type(item).__name__})')
            if isinstance(item, h5py.Group):
                self._print_hdf5_structure(item, indent+1)
    
    def _extract_frames_from_hdf5(self, hdf5_file, metadata):
        '''Extract frames from an HDF5 file based on its structure.'''
        frames = []
        
        # Structure observed in the output: /Frame_frame_XXXX/array
        frame_keys = [k for k in hdf5_file.keys() if k.startswith('Frame_frame_')]
        
        if frame_keys:
            # Sort by frame number to maintain temporal order
            sorted_frame_keys = sorted(frame_keys, key=lambda k: int(k.split('_')[-1]))
            
            for key in sorted_frame_keys:
                if 'array' in hdf5_file[key]:
                    frames.append(np.array(hdf5_file[key]['array']))
            
            return frames
        
        # If the above structure isn't found, try alternative structures:
        
        # Structure 1: Direct keys as frame indices
        if all(key.isdigit() for key in list(hdf5_file.keys())[:5] if key not in ['FrameIndexes']):
            digit_keys = [k for k in hdf5_file.keys() if k.isdigit()]
            for key in sorted(digit_keys, key=int):
                frames.append(np.array(hdf5_file[key]))
        
        # Structure 2: stream_0/recording_1/frame_XXXXXXXX
        elif 'stream_0' in hdf5_file and 'recording_1' in hdf5_file['stream_0']:
            recording_path = 'stream_0/recording_1'
            # Sample frames (20 for most, 80 for print attacks)
            frame_count = 80 if metadata and metadata.get('type_id') == '01' else 20
            frame_keys = sorted(list(hdf5_file[recording_path].keys()))
            total_frames = len(frame_keys)
            
            if total_frames > 0:
                indices = np.linspace(0, total_frames-1, min(frame_count, total_frames), dtype=int)
                for idx in indices:
                    frame_key = frame_keys[idx]
                    frames.append(np.array(hdf5_file[recording_path][frame_key]))
        
        # Structure 3: annotations structure
        elif 'annotations' in hdf5_file:
            for key in sorted(hdf5_file['annotations'].keys()):
                frames.append(np.array(hdf5_file['annotations'][key]))
        
        # Structure 4: Check if FrameIndexes points to valid frames
        elif 'FrameIndexes' in hdf5_file:
            # The FrameIndexes group might contain references to the actual frames
            frame_indices = []
            for i in range(20):  # From output, we saw indices 0-19
                if str(i) in hdf5_file['FrameIndexes']:
                    idx = int(np.array(hdf5_file['FrameIndexes'][str(i)]))
                    frame_indices.append(idx)
            
            # Now try to find frames corresponding to these indices
            for frame_key in hdf5_file.keys():
                if frame_key != 'FrameIndexes' and isinstance(hdf5_file[frame_key], h5py.Group):
                    if 'array' in hdf5_file[frame_key]:
                        frames.append(np.array(hdf5_file[frame_key]['array']))
        
        return frames
    
    def load_data(self, max_samples=None, include_attacks=True, include_bona_fide=True):
        '''
        Load data from the dataset.
        
        Args:
            max_samples (int): Maximum number of samples to load (None for all)
            include_attacks (bool): Whether to include attack samples
            include_bona_fide (bool): Whether to include bona fide samples
            
        Returns:
            list: List of loaded samples
        '''
        samples = []
        
        files_to_load = []
        if include_bona_fide:
            files_to_load.extend(self.bona_fide_files)
        if include_attacks:
            files_to_load.extend(self.attack_files)
        
        if max_samples is not None:
            files_to_load = files_to_load[:max_samples]
        
        for file_info in tqdm(files_to_load, desc=f'Loading {self.subset} data'):
            sample = self.load_sample(file_info)
            if sample:
                samples.append(sample)
        
        return samples
    
    def get_statistics(self):
        '''Get statistics about the loaded protocol.'''
        stats = {
            'total_files': len(self.bona_fide_files) + len(self.attack_files),
            'bona_fide_files': len(self.bona_fide_files),
            'attack_files': len(self.attack_files),
            'subset': self.subset,
            'protocol': self.protocol
        }
        return stats

# Custom PyTorch Dataset for VFPAD
class VFPADTorchDataset(Dataset):
    def __init__(self, samples, transform=None, frames_per_sample=5):
        '''
        Args:
            samples: List of samples from VFPADDataset.load_data()
            transform: Torchvision transforms for preprocessing
            frames_per_sample: Number of frames to use from each video
        '''
        self.samples = samples
        self.transform = transform
        self.frames_per_sample = frames_per_sample
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = sample['frames']
        label = 1 if sample['is_attack'] else 0  # 1 for attack, 0 for bona fide
        
        # If we have no frames, return a zero tensor with the right shape
        if not frames or len(frames) == 0:
            # Return a black image of the expected size (224x224)
            img_tensor = torch.zeros(3, 224, 224)
            return img_tensor, label
        
        # Select frames uniformly across the video
        if len(frames) >= self.frames_per_sample:
            indices = np.linspace(0, len(frames) - 1, self.frames_per_sample, dtype=int)
            selected_frames = [frames[i] for i in indices]
        else:
            # If we have fewer frames than requested, duplicate the last frame
            selected_frames = frames + [frames[-1]] * (self.frames_per_sample - len(frames))
        
        # Process frames
        processed_frames = []
        for frame in selected_frames:
            # Ensure frame has proper dimensions and type
            if frame.ndim == 2:  # Grayscale
                frame = np.stack([frame] * 3, axis=2)  # Convert to RGB
            elif frame.ndim == 3 and frame.shape[2] == 1:  # Grayscale with channel dim
                frame = np.concatenate([frame] * 3, axis=2)  # Convert to RGB
            elif frame.ndim == 3 and frame.shape[2] > 3:  # More than 3 channels
                frame = frame[:, :, :3]  # Take first 3 channels
            
            # Ensure frame is in uint8 range (0-255)
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            if self.transform:
                frame = self.transform(frame)
            
            processed_frames.append(frame)
        
        # Stack frames along batch dimension
        frame_tensor = torch.stack(processed_frames)
        
        # For this implementation, return the first frame and label
        # You could return all frames for sequence model approaches
        return frame_tensor[0], label


# MobileNetV3 with attention for PAD
class MobileNetPAD(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(MobileNetPAD, self).__init__()
        # Load MobileNetV3 Small
        self.mobilenet = mobilenet_v3_small(pretrained=pretrained)
        
        # Replace the classifier layer
        in_features = self.mobilenet.classifier[0].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.mobilenet(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    '''Training function for the PAD model'''
    best_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.float().to(device).unsqueeze(1)  # Binary classification
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                # Store labels and scores for AUC calculation
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(torch.sigmoid(outputs).cpu().numpy())
        
        # Calculate validation metrics
        val_loss = running_loss / len(val_loader.dataset)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        val_auc = auc(fpr, tpr)
        
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        print(f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model_mobilenetv3_pad.pth')
            print(f'New best model saved with AUC: {val_auc:.4f}')
    
    return model, history


def plot_training_history(history):
    '''Plot training and validation metrics'''
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot AUC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_auc'], 'g-', label='Validation AUC')
    plt.title('Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset loader
    dataset_path = 'VFPAD'
    dataset = VFPADDataset(dataset_path, protocol='grandtest', subset='train')
    
    # Load data
    all_samples = dataset.load_data()
    print(f'Loaded {len(all_samples)} samples')
    
    # Split into train and validation sets
    train_samples, val_samples = train_test_split(
        all_samples, test_size=0.2, random_state=42,
        stratify=[s['is_attack'] for s in all_samples]
    )
    
    print(f'Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}')
    
    # Data transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create PyTorch datasets
    train_dataset = VFPADTorchDataset(train_samples, transform=transform)
    val_dataset = VFPADTorchDataset(val_samples, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create model
    model = MobileNetPAD(pretrained=True).to(device)
    
    # Loss function, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        num_epochs=10, device=device
    )
    
    # Plot results
    plot_training_history(history)
    
    # Evaluate on test set (if available)
    test_dataset = VFPADDataset(dataset_path, protocol='grandtest', subset='eval')
    test_samples = test_dataset.load_data()
    
    if test_samples:
        print(f'Evaluating on {len(test_samples)} test samples')
        
        test_torch_dataset = VFPADTorchDataset(test_samples, transform=transform)
        test_loader = DataLoader(test_torch_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Load best model
        model.load_state_dict(torch.load('best_model_mobilenetv3_pad.pth'))
        model.eval()
        
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing'):
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(torch.sigmoid(outputs).cpu().numpy())
        
        # Calculate test metrics
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        test_auc = auc(fpr, tpr)
        
        print(f'Test AUC: {test_auc:.4f}')
        
        # Find the threshold for Equal Error Rate (EER)
        fnr = 1 - tpr
        eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
        eer_threshold = thresholds[eer_threshold_idx]
        
        print(f'EER: {eer:.4f} at threshold: {eer_threshold:.4f}')
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {test_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curve.png')
        plt.show()