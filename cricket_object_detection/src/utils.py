"""
Common utility functions for cricket object detection.
"""

import os
import json
import pickle
from typing import List, Tuple, Dict, Any


# Class labels
CLASS_LABELS = {
    0: 'no_object',
    1: 'ball',
    2: 'bat',
    3: 'stump'
}

# Reverse mapping
LABEL_TO_CLASS = {v: k for k, v in CLASS_LABELS.items()}

# Image dimensions
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
GRID_ROWS = 8
GRID_COLS = 8
CELL_WIDTH = IMAGE_WIDTH // GRID_COLS  # 100 pixels
CELL_HEIGHT = IMAGE_HEIGHT // GRID_ROWS  # 75 pixels
TOTAL_CELLS = GRID_ROWS * GRID_COLS  # 64 cells


def get_grid_cell_bounds(cell_index: int) -> Tuple[int, int, int, int]:
    """
    Get the pixel bounds of a grid cell.
    
    Args:
        cell_index: Cell index (1-64, as in c01-c64)
    
    Returns:
        Tuple of (x_start, y_start, x_end, y_end)
    """
    if cell_index < 1 or cell_index > TOTAL_CELLS:
        raise ValueError(f"Cell index must be between 1 and {TOTAL_CELLS}")
    
    # Convert to 0-indexed
    idx = cell_index - 1
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    
    x_start = col * CELL_WIDTH
    y_start = row * CELL_HEIGHT
    x_end = x_start + CELL_WIDTH
    y_end = y_start + CELL_HEIGHT
    
    return x_start, y_start, x_end, y_end


def get_cell_from_position(x: int, y: int) -> int:
    """
    Get the cell index (1-64) from pixel coordinates.
    
    Args:
        x: X coordinate (0-799)
        y: Y coordinate (0-599)
    
    Returns:
        Cell index (1-64)
    """
    col = min(x // CELL_WIDTH, GRID_COLS - 1)
    row = min(y // CELL_HEIGHT, GRID_ROWS - 1)
    return row * GRID_COLS + col + 1


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: The trained model object
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the model file
    
    Returns:
        The loaded model object
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def save_json(data: Dict, filepath: str) -> None:
    """Save data as JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files in a directory.
    
    Args:
        directory: Path to the directory
        extensions: List of valid file extensions (default: common image formats)
    
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    image_files = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(directory, filename))
    
    return sorted(image_files)


def create_csv_header() -> str:
    """Create the CSV header row."""
    columns = ['ImageFileName', 'TrainOrTest']
    columns.extend([f'c{i:02d}' for i in range(1, TOTAL_CELLS + 1)])
    return ','.join(columns)


def create_csv_row(filename: str, train_or_test: str, predictions: List[int]) -> str:
    """
    Create a CSV row for predictions.
    
    Args:
        filename: Image filename
        train_or_test: 'Train' or 'Test'
        predictions: List of 64 predictions (0-3)
    
    Returns:
        CSV row string
    """
    if len(predictions) != TOTAL_CELLS:
        raise ValueError(f"Expected {TOTAL_CELLS} predictions, got {len(predictions)}")
    
    row_data = [filename, train_or_test]
    row_data.extend([str(p) for p in predictions])
    return ','.join(row_data)
