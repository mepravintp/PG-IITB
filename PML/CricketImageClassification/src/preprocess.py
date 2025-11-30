"""
Image preprocessing module for cricket image classification.
Handles image resizing, grid division, and data preparation.
"""

import numpy as np
from PIL import Image
import os
from typing import Tuple, List, Optional


# Constants
TARGET_WIDTH = 800
TARGET_HEIGHT = 600
GRID_ROWS = 8
GRID_COLS = 8
CELL_WIDTH = TARGET_WIDTH // GRID_COLS  # 100 pixels
CELL_HEIGHT = TARGET_HEIGHT // GRID_ROWS  # 75 pixels


def validate_image_resolution(image: Image.Image) -> bool:
    """
    Check if image resolution is at least 800x600.
    Images below this resolution should not be used.
    
    Args:
        image: PIL Image object
        
    Returns:
        bool: True if resolution is valid, False otherwise
    """
    width, height = image.size
    return width >= TARGET_WIDTH and height >= TARGET_HEIGHT


def resize_image(image: Image.Image) -> Image.Image:
    """
    Resize image to 800x600 pixels while maintaining 4:3 aspect ratio.
    
    Args:
        image: PIL Image object
        
    Returns:
        Image: Resized image (800x600)
    """
    # Calculate aspect ratio of original image
    orig_width, orig_height = image.size
    target_ratio = TARGET_WIDTH / TARGET_HEIGHT  # 4:3
    orig_ratio = orig_width / orig_height
    
    if orig_ratio > target_ratio:
        # Image is wider than 4:3, crop width
        new_width = int(orig_height * target_ratio)
        left = (orig_width - new_width) // 2
        image = image.crop((left, 0, left + new_width, orig_height))
    elif orig_ratio < target_ratio:
        # Image is taller than 4:3, crop height
        new_height = int(orig_width / target_ratio)
        top = (orig_height - new_height) // 2
        image = image.crop((0, top, orig_width, top + new_height))
    
    # Resize to target dimensions
    return image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)


def divide_into_grid(image: np.ndarray) -> List[np.ndarray]:
    """
    Divide an 800x600 image into 8x8 grid (64 cells).
    Each cell is 100x75 pixels.
    
    Args:
        image: numpy array of shape (600, 800, 3)
        
    Returns:
        List of 64 cell arrays, row-major order (c01, c02, ..., c64)
    """
    cells = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y_start = row * CELL_HEIGHT
            y_end = y_start + CELL_HEIGHT
            x_start = col * CELL_WIDTH
            x_end = x_start + CELL_WIDTH
            cell = image[y_start:y_end, x_start:x_end]
            cells.append(cell)
    return cells


def get_cell_index(row: int, col: int) -> int:
    """
    Get cell index (1-based) from row and column (0-based).
    
    Args:
        row: Row index (0-7)
        col: Column index (0-7)
        
    Returns:
        Cell index (1-64)
    """
    return row * GRID_COLS + col + 1


def get_row_col(cell_index: int) -> Tuple[int, int]:
    """
    Get row and column (0-based) from cell index (1-based).
    
    Args:
        cell_index: Cell index (1-64)
        
    Returns:
        Tuple of (row, col) (0-based)
    """
    idx = cell_index - 1
    return idx // GRID_COLS, idx % GRID_COLS


def load_and_preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image, validate, resize, and convert to numpy array.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array of shape (600, 800, 3) or None if invalid
    """
    try:
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Validate resolution
        if not validate_image_resolution(image):
            print(f"Warning: Image {image_path} has resolution below 800x600. Skipping.")
            return None
        
        # Resize to 800x600
        image = resize_image(image)
        
        # Convert to numpy array
        return np.array(image)
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_images_from_directory(directory: str) -> List[Tuple[str, np.ndarray]]:
    """
    Load all valid images from a directory.
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        List of tuples (filename, image_array)
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = []
    
    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_path = os.path.join(directory, filename)
            image_array = load_and_preprocess_image(image_path)
            if image_array is not None:
                images.append((filename, image_array))
    
    return images


def visualize_grid(image: np.ndarray, labels: Optional[List[int]] = None) -> np.ndarray:
    """
    Draw grid lines on image for visualization.
    Optionally overlay cell labels.
    
    Args:
        image: numpy array of shape (600, 800, 3)
        labels: Optional list of 64 labels (0-3) for each cell
        
    Returns:
        Image with grid overlay
    """
    # Make a copy to avoid modifying original
    result = image.copy()
    
    # Draw vertical lines
    for col in range(1, GRID_COLS):
        x = col * CELL_WIDTH
        result[:, x-1:x+1] = [255, 0, 0]  # Red lines
    
    # Draw horizontal lines
    for row in range(1, GRID_ROWS):
        y = row * CELL_HEIGHT
        result[y-1:y+1, :] = [255, 0, 0]  # Red lines
    
    return result
