import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tqdm import tqdm

def convert_all_hdf5_to_video(dataset_path, output_path, fps=30):
    """
    Convert all HDF5 frames from all files in directory to a single MP4 video
    
    Args:
        dataset_path (str): Path to directory containing HDF5 files
        output_path (str): Path to save MP4 file
        fps (int): Frames per second for output video
    """
    video_writer = None
    
    # Process each HDF5 file in the directory
    for root, dirs, files in os.walk(dataset_path):
        # Sort files for consistent ordering
        files = sorted([f for f in files if f.endswith('.hdf5')])
        
        for file in files:
            file_path = os.path.join(root, file)
            print(f"\nProcessing file: {file}")
            
            with h5py.File(file_path, 'r') as f:
                # Get all frame keys
                frame_keys = sorted([k for k in f.keys() if k.startswith('Frame_frame_')])
                if not frame_keys:
                    continue
                
                # Initialize video writer with first frame if not already done
                if video_writer is None:
                    first_frame = np.array(f[f"{frame_keys[0]}/array"][:])
                    if first_frame.dtype != np.uint8:
                        first_frame = (first_frame * 255).astype(np.uint8)
                    if len(first_frame.shape) == 3 and first_frame.shape[0] == 3:
                        first_frame = np.transpose(first_frame, (1, 2, 0))
                    
                    height, width = first_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Process each frame
                for frame_key in tqdm(frame_keys, desc=f"Processing {file}"):
                    # Load frame
                    frame = np.array(f[f"{frame_key}/array"][:])
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    if len(frame.shape) == 3 and frame.shape[0] == 3:
                        frame = np.transpose(frame, (1, 2, 0))
                    
                    # Convert to BGR for OpenCV
                    if len(frame.shape) == 2:  # Grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:  # RGB to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Write frame
                    video_writer.write(frame)
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\nCombined video saved to: {output_path}")
    else:
        print("No frames were found to process")

if __name__ == "__main__":
    # Example usage with a presentation attack folder
    dataset_path = "VFPAD/data/pa/pa_0001"
    output_path = "combined_video.mp4"
    
    # Convert all frames to a single video
    convert_all_hdf5_to_video(dataset_path, output_path)