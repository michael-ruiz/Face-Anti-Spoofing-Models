import cv2
import numpy as np
from typing import List, Tuple

def get_patches(image: np.ndarray, patch_size: int) -> List[np.ndarray]:
    """
    Split an image into basic square patches.
    
    Args:
        image (np.ndarray): Input image
        patch_size (int): Size of square patches (patch_size x patch_size)
    
    Returns:
        List[np.ndarray]: List of image patches
    """
    height, width = image.shape[:2]
    
    # Calculate number of patches in each dimension
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    patches = []
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Extract patch
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size
            
            patch = image[y_start:y_end, x_start:x_end]
            patches.append(patch)
    
    return patches

def create_combined_visualization(original_image: np.ndarray, patches: List[np.ndarray], grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Create a combined visualization with original image and patches in a square layout.
    
    Args:
        original_image (np.ndarray): Original input image
        patches (List[np.ndarray]): List of image patches
        grid_size (tuple): Size of patches visualization grid (rows, cols)
    
    Returns:
        np.ndarray: Combined visualization
    """
    rows, cols = grid_size
    patch_height, patch_width = patches[0].shape[:2]
    
    # Create patches grid
    patches_grid = np.zeros((rows * patch_height, cols * patch_width, 3), dtype=np.uint8)
    
    for idx, patch in enumerate(patches):
        if idx >= rows * cols:
            break
            
        i = idx // cols
        j = idx % cols
        
        y_start = i * patch_height
        y_end = y_start + patch_height
        x_start = j * patch_width
        x_end = x_start + patch_width
        
        patches_grid[y_start:y_end, x_start:x_end] = patch

    # Resize original image to match height of patches grid
    target_height = patches_grid.shape[0]
    aspect_ratio = original_image.shape[1] / original_image.shape[0]
    target_width = int(target_height * aspect_ratio)
    resized_original = cv2.resize(original_image, (target_width, target_height))

    # Create the final combined image
    combined_width = resized_original.shape[1] + patches_grid.shape[1]
    combined_image = np.zeros((target_height, combined_width, 3), dtype=np.uint8)
    
    # Place original image on the left
    combined_image[:, :target_width] = resized_original
    # Place patches grid on the right
    combined_image[:, target_width:] = patches_grid
    
    return combined_image

# Example usage
if __name__ == "__main__":
    # Load an example image
    image_path = "./test_img/560253.png"
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image")
    else:
        # Get patches
        patch_size = 32  # Size for square patches
        patches = get_patches(image, patch_size)
        
        # Create combined visualization
        grid_size = (8, 8)
        combined_vis = create_combined_visualization(image, patches, grid_size)
        
        # Display result in a single window
        cv2.imshow("Original Image and Patches", combined_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()