"""
Utility functions for cricket image classification.
Handles CSV output, annotations, and general utilities.
"""

import os
import json
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np


def save_predictions_to_csv(predictions: Dict[str, Tuple[str, List[int]]], 
                            output_path: str) -> None:
    """
    Save predictions to CSV file in the required format.
    
    Args:
        predictions: Dictionary mapping image filename to (train/test, list of 64 cell predictions)
        output_path: Path to save the CSV file
    """
    # Create header
    header = ['ImageFileName', 'TrainOrTest']
    for i in range(1, 65):
        header.append(f'c{i:02d}')
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for filename, (data_type, cell_predictions) in predictions.items():
            row = [filename, data_type]
            row.extend(cell_predictions)
            writer.writerow(row)
    
    print(f"Predictions saved to {output_path}")


def load_predictions_from_csv(csv_path: str) -> Dict[str, Tuple[str, List[int]]]:
    """
    Load predictions from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Dictionary mapping image filename to (train/test, list of 64 cell predictions)
    """
    predictions = {}
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        
        for row in reader:
            filename = row[0]
            data_type = row[1]
            cell_preds = [int(x) for x in row[2:]]
            predictions[filename] = (data_type, cell_preds)
    
    return predictions


def save_annotations(annotations: Dict[str, List[int]], output_path: str) -> None:
    """
    Save annotations to JSON file.
    
    Args:
        annotations: Dictionary mapping image filename to list of 64 cell labels
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Annotations saved to {output_path}")


def load_annotations(json_path: str) -> Dict[str, List[int]]:
    """
    Load annotations from JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary mapping image filename to list of 64 cell labels
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def create_sample_annotations(image_dir: str, output_path: str) -> None:
    """
    Create sample annotations file for manual labeling.
    
    Args:
        image_dir: Directory containing images
        output_path: Path to save the annotations template
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    annotations = {}
    
    for filename in os.listdir(image_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            # Initialize all cells as no_object (0)
            annotations[filename] = [0] * 64
    
    save_annotations(annotations, output_path)
    print(f"Sample annotations template created at {output_path}")
    print(f"Found {len(annotations)} images. Please manually edit the annotations.")


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print a formatted confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    from .model import LABEL_NAMES
    
    print("\nConfusion Matrix:")
    print("-" * 60)
    print(f"{'':12s} |", end="")
    for i in range(4):
        print(f" {LABEL_NAMES[i][:10]:>10s} |", end="")
    print()
    print("-" * 60)
    
    for i in range(4):
        print(f" {LABEL_NAMES[i][:10]:10s} |", end="")
        for j in range(4):
            count = np.sum((y_true == i) & (y_pred == j))
            print(f" {count:10d} |", end="")
        print()
    print("-" * 60)


def get_cell_coordinates(cell_index: int) -> Tuple[int, int, int, int]:
    """
    Get pixel coordinates for a cell.
    
    Args:
        cell_index: Cell index (1-64)
        
    Returns:
        Tuple of (x_start, y_start, x_end, y_end)
    """
    CELL_WIDTH = 100
    CELL_HEIGHT = 75
    GRID_COLS = 8
    
    idx = cell_index - 1
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    
    x_start = col * CELL_WIDTH
    y_start = row * CELL_HEIGHT
    x_end = x_start + CELL_WIDTH
    y_end = y_start + CELL_HEIGHT
    
    return (x_start, y_start, x_end, y_end)


def summarize_predictions(predictions: Dict[str, Tuple[str, List[int]]]) -> None:
    """
    Print summary statistics of predictions.
    
    Args:
        predictions: Dictionary of predictions
    """
    from .model import LABEL_NAMES
    
    total_cells = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    train_count = 0
    test_count = 0
    
    for filename, (data_type, cell_preds) in predictions.items():
        if data_type.lower() == 'train':
            train_count += 1
        else:
            test_count += 1
        
        for pred in cell_preds:
            class_counts[pred] += 1
            total_cells += 1
    
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Total images: {len(predictions)}")
    print(f"  Training: {train_count}")
    print(f"  Testing: {test_count}")
    print(f"\nTotal cells: {total_cells}")
    print("\nClass distribution:")
    for class_id, count in class_counts.items():
        percentage = (count / total_cells * 100) if total_cells > 0 else 0
        print(f"  {LABEL_NAMES[class_id]}: {count} ({percentage:.2f}%)")
    print("=" * 50)


def validate_dataset_structure(base_dir: str) -> bool:
    """
    Validate that the dataset directory structure is correct.
    
    Args:
        base_dir: Base directory of the dataset
        
    Returns:
        True if structure is valid, False otherwise
    """
    expected_dirs = [
        'data/train/ball',
        'data/train/bat',
        'data/train/stump',
        'data/train/no_object',
        'data/test/ball',
        'data/test/bat',
        'data/test/stump',
        'data/test/no_object'
    ]
    
    valid = True
    for dir_path in expected_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"Missing directory: {full_path}")
            valid = False
    
    return valid


def count_images(base_dir: str) -> Dict[str, int]:
    """
    Count images in each category.
    
    Args:
        base_dir: Base directory of the dataset
        
    Returns:
        Dictionary with counts for each category
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    counts = {}
    
    for split in ['train', 'test']:
        for category in ['ball', 'bat', 'stump', 'no_object']:
            dir_path = os.path.join(base_dir, 'data', split, category)
            if os.path.exists(dir_path):
                count = sum(1 for f in os.listdir(dir_path) 
                           if os.path.splitext(f)[1].lower() in valid_extensions)
                counts[f"{split}/{category}"] = count
            else:
                counts[f"{split}/{category}"] = 0
    
    return counts
