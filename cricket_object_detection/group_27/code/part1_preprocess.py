"""
Image preprocessing utilities for cricket object detection.
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import src.utils

from src.utils import IMAGE_WIDTH, IMAGE_HEIGHT



def validate_image_size(image: Image.Image) -> bool:
    """
    Validate that an image meets the minimum size requirements.
    
    Images must be at least 800x600 pixels (we don't upscale).
    
    Args:
        image: PIL Image object
    
    Returns:
        True if the image meets size requirements, False otherwise
    """
    width, height = image.size
    return width >= IMAGE_WIDTH and height >= IMAGE_HEIGHT


def calculate_aspect_ratio(width: int, height: int) -> Tuple[int, int]:
    """
    Calculate the simplified aspect ratio.
    
    Args:
        width: Image width
        height: Image height
    
    Returns:
        Tuple of (width_ratio, height_ratio)
    """
    from math import gcd
    divisor = gcd(width, height)
    return width // divisor, height // divisor


def is_aspect_ratio_4_3(width: int, height: int, tolerance: float = 0.05) -> bool:
    """
    Check if the aspect ratio is approximately 4:3.
    
    Args:
        width: Image width
        height: Image height
        tolerance: Allowed deviation from 4:3 ratio (default 5%)
    
    Returns:
        True if aspect ratio is approximately 4:3
    """
    target_ratio = 4 / 3
    actual_ratio = width / height
    return abs(actual_ratio - target_ratio) / target_ratio <= tolerance


def resize_image(image: Image.Image, target_width: int = IMAGE_WIDTH, 
                 target_height: int = IMAGE_HEIGHT) -> Image.Image:
    """
    Resize an image to the target dimensions.
    
    Uses LANCZOS resampling for high-quality downscaling.
    
    Args:
        image: PIL Image object
        target_width: Target width (default 800)
        target_height: Target height (default 600)
    
    Returns:
        Resized PIL Image object
    """
    return image.resize((target_width, target_height), Image.LANCZOS)


def crop_to_aspect_ratio(image: Image.Image, aspect_width: int = 4, 
                          aspect_height: int = 3) -> Image.Image:
    """
    Crop an image to the specified aspect ratio (center crop).
    
    Args:
        image: PIL Image object
        aspect_width: Width aspect ratio
        aspect_height: Height aspect ratio
    
    Returns:
        Cropped PIL Image object
    """
    width, height = image.size
    target_ratio = aspect_width / aspect_height
    current_ratio = width / height
    
    if abs(current_ratio - target_ratio) < 0.001:
        # Already correct aspect ratio
        return image
    
    if current_ratio > target_ratio:
        # Image is too wide, crop width
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))
    else:
        # Image is too tall, crop height
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        return image.crop((0, top, width, top + new_height))


def preprocess_image(image_path: str, output_path: Optional[str] = None) -> Optional[Image.Image]:
    """
    Preprocess a single image: validate, crop to 4:3, and resize to 800x600.
    
    Args:
        image_path: Path to the input image
        output_path: Optional path to save the processed image
    
    Returns:
        Processed PIL Image object, or None if image doesn't meet requirements
    """
    try:
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Check minimum size (before any cropping)
        width, height = image.size
        
        # For non-4:3 images, we need to check if we can get 800x600 after cropping
        target_ratio = 4 / 3
        current_ratio = width / height
        
        if current_ratio > target_ratio:
            # Will crop width - check height is sufficient
            if height < IMAGE_HEIGHT:
                return None
            potential_width = int(height * target_ratio)
            if potential_width < IMAGE_WIDTH:
                return None
        else:
            # Will crop height - check width is sufficient
            if width < IMAGE_WIDTH:
                return None
            potential_height = int(width / target_ratio)
            if potential_height < IMAGE_HEIGHT:
                return None
        
        # Crop to 4:3 aspect ratio
        image = crop_to_aspect_ratio(image)
        
        # Validate final size
        if not validate_image_size(image):
            return None
        
        # Resize to target dimensions
        image = resize_image(image)
        
        # Save if output path provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            image.save(output_path, quality=95)
        
        return image
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def preprocess_directory(input_dir: str, output_dir: str, 
                          extensions: list = None) -> Tuple[int, int]:
    """
    Preprocess all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        extensions: List of valid file extensions
    
    Returns:
        Tuple of (processed_count, skipped_count)
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed = 0
    skipped = 0
    
    for filename in os.listdir(input_dir):
        if not any(filename.lower().endswith(ext) for ext in extensions):
            continue
        
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        result = preprocess_image(input_path, output_path)
        
        if result is not None:
            processed += 1
        else:
            skipped += 1
            print(f"Skipped: {filename}")
    
    return processed, skipped


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a numpy array.
    
    Args:
        image: PIL Image object
    
    Returns:
        Numpy array of shape (height, width, 3) with dtype uint8
    """
    return np.array(image, dtype=np.uint8)


def numpy_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert a numpy array to a PIL Image.
    
    Args:
        array: Numpy array of shape (height, width, 3)
    
    Returns:
        PIL Image object
    """
    return Image.fromarray(array.astype(np.uint8))


def extract_grid_cell(image: np.ndarray, cell_index: int) -> np.ndarray:
    """
    Extract a single grid cell from an image.
    
    Args:
        image: Numpy array of shape (600, 800, 3)
        cell_index: Cell index (1-64)
    
    Returns:
        Numpy array of shape (75, 100, 3) for the cell
    """
    from src.utils import get_grid_cell_bounds
    
    x_start, y_start, x_end, y_end = get_grid_cell_bounds(cell_index)
    return image[y_start:y_end, x_start:x_end]


def split_image_to_grid(image: np.ndarray) -> list:
    """
    Split an image into 64 grid cells.
    
    Args:
        image: Numpy array of shape (600, 800, 3)
    
    Returns:
        List of 64 numpy arrays, each of shape (75, 100, 3)
    """
    try:
        from src.utils import TOTAL_CELLS
    except ImportError:
        from utils import TOTAL_CELLS
    
    cells = []
    for i in range(1, TOTAL_CELLS + 1):
        cell = extract_grid_cell(image, i)
        cells.append(cell)
    
    return cells
