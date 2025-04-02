import h5py
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_hdf5_image(file_path, frame_index=0):
    """
    Load and visualize an HDF5 image from the VFPAD dataset
    
    Args:
        file_path (str): Path to HDF5 file
        frame_index (int): Index of frame to visualize (default: 0)
    """
    with h5py.File(file_path, 'r') as f:
        # Print available keys and structure
        print("\nHDF5 file structure:")
        f.visit(lambda name: print(name))
        
        # Get all frame keys (excluding FrameIndexes)
        frame_keys = sorted([k for k in f.keys() if k.startswith('Frame_frame_')])
        if not frame_keys:
            raise KeyError("No frames found in HDF5 file")
        
        # Select frame to visualize
        frame_key = frame_keys[frame_index]
        print(f"\nLoading frame: {frame_key}")
        
        # Load the image data - now accessing the 'array' subgroup
        frame_data = f[f"{frame_key}/array"][:]
        print(f"Dataset info: shape={frame_data.shape}, dtype={frame_data.dtype}")
        
        # Load the actual image data
        image = np.array(frame_data)
        
        # Convert to appropriate format if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # If RGB, transpose to correct format (H,W,C)
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Display the image
        plt.figure(figsize=(10, 10))
        if len(image.shape) == 2 or image.shape[-1] == 1:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.axis('off')
        plt.title(f"Frame {frame_key} from {os.path.basename(file_path)}")
        plt.show()
        
        # Print image information
        print(f"Image shape: {image.shape}")
        print(f"Data type: {image.dtype}")
        print(f"Value range: [{image.min()}, {image.max()}]")

if __name__ == "__main__":
    # Example usage with a bona fide image
    dataset_path = "VFPAD/data/pa/pa_0104"
    
    # Get the first HDF5 file in the directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.hdf5'):
                file_path = os.path.join(root, file)
                print(f"Loading file: {file_path}")
                
                # Visualize first and last frames
                visualize_hdf5_image(file_path, frame_index=0)  # First frame
                break