"""
Annotation utilities for grid-based object labeling.
"""

import os
import json
from typing import Dict, List, Optional
from PIL import Image
import numpy as np

from .utils import (
    GRID_ROWS, GRID_COLS, TOTAL_CELLS, IMAGE_WIDTH, IMAGE_HEIGHT,
    CELL_WIDTH, CELL_HEIGHT, CLASS_LABELS, LABEL_TO_CLASS,
    get_grid_cell_bounds, save_json, load_json
)


def create_empty_annotation() -> List[int]:
    """
    Create an empty annotation with all cells marked as no_object (0).
    
    Returns:
        List of 64 zeros
    """
    return [0] * TOTAL_CELLS


def create_annotation(image_filename: str, cell_labels: Dict[int, int] = None) -> Dict:
    """
    Create an annotation record for an image.
    
    Args:
        image_filename: Name of the image file
        cell_labels: Dictionary mapping cell indices (1-64) to class labels (0-3)
    
    Returns:
        Annotation dictionary
    """
    labels = create_empty_annotation()
    
    if cell_labels:
        for cell_idx, label in cell_labels.items():
            if 1 <= cell_idx <= TOTAL_CELLS and 0 <= label <= 3:
                labels[cell_idx - 1] = label
    
    return {
        'image_filename': image_filename,
        'labels': labels,
        'grid_size': [GRID_ROWS, GRID_COLS],
        'image_size': [IMAGE_WIDTH, IMAGE_HEIGHT]
    }


def save_annotation(annotation: Dict, filepath: str) -> None:
    """
    Save annotation to a JSON file.
    
    Args:
        annotation: Annotation dictionary
        filepath: Path to save the annotation
    """
    save_json(annotation, filepath)
    print(f"Annotation saved to {filepath}")


def load_annotation(filepath: str) -> Dict:
    """
    Load annotation from a JSON file.
    
    Args:
        filepath: Path to the annotation file
    
    Returns:
        Annotation dictionary
    """
    return load_json(filepath)


def save_annotations_batch(annotations: List[Dict], filepath: str) -> None:
    """
    Save multiple annotations to a single JSON file.
    
    Args:
        annotations: List of annotation dictionaries
        filepath: Path to save the annotations
    """
    save_json(annotations, filepath)
    print(f"Saved {len(annotations)} annotations to {filepath}")


def load_annotations_batch(filepath: str) -> List[Dict]:
    """
    Load multiple annotations from a JSON file.
    
    Args:
        filepath: Path to the annotations file
    
    Returns:
        List of annotation dictionaries
    """
    return load_json(filepath)


def annotation_summary(annotations: List[Dict]) -> Dict:
    """
    Generate a summary of annotation statistics.
    
    Args:
        annotations: List of annotation dictionaries
    
    Returns:
        Summary dictionary with counts per class
    """
    total_cells = len(annotations) * TOTAL_CELLS
    class_counts = {label: 0 for label in CLASS_LABELS.values()}
    
    for ann in annotations:
        for label in ann['labels']:
            class_counts[CLASS_LABELS[label]] += 1
    
    return {
        'total_images': len(annotations),
        'total_cells': total_cells,
        'class_counts': class_counts,
        'class_percentages': {
            k: f"{100 * v / total_cells:.2f}%" 
            for k, v in class_counts.items()
        }
    }


def visualize_annotation(image: np.ndarray, labels: List[int], 
                          output_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize annotations on an image by drawing grid and labels.
    
    Args:
        image: RGB image array of shape (600, 800, 3)
        labels: List of 64 labels (0-3)
        output_path: Optional path to save the visualization
    
    Returns:
        Annotated image array
    """
    from PIL import ImageDraw, ImageFont
    
    # Create a copy
    img = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    
    # Colors for each class
    colors = {
        0: (128, 128, 128),  # Gray for no_object
        1: (255, 0, 0),       # Red for ball
        2: (0, 255, 0),       # Green for bat
        3: (0, 0, 255)        # Blue for stump
    }
    
    # Draw grid and labels
    for i, label in enumerate(labels):
        cell_idx = i + 1
        x1, y1, x2, y2 = get_grid_cell_bounds(cell_idx)
        
        color = colors[label]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label text
        text = str(label)
        draw.text((x1 + 5, y1 + 5), text, fill=color)
    
    result = np.array(img)
    
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        img.save(output_path)
        print(f"Visualization saved to {output_path}")
    
    return result


def get_annotation_for_image(image_filename: str, 
                              annotations_file: str) -> Optional[List[int]]:
    """
    Get the annotation labels for a specific image.
    
    Args:
        image_filename: Name of the image file
        annotations_file: Path to the annotations JSON file
    
    Returns:
        List of 64 labels, or None if image not found
    """
    annotations = load_annotations_batch(annotations_file)
    
    for ann in annotations:
        if ann['image_filename'] == image_filename:
            return ann['labels']
    
    return None


def merge_annotations(annotations_list: List[List[Dict]]) -> List[Dict]:
    """
    Merge multiple lists of annotations into one.
    
    Args:
        annotations_list: List of annotation lists to merge
    
    Returns:
        Merged list of annotations
    """
    merged = []
    seen_filenames = set()
    
    for annotations in annotations_list:
        for ann in annotations:
            filename = ann['image_filename']
            if filename not in seen_filenames:
                merged.append(ann)
                seen_filenames.add(filename)
    
    return merged


def create_sample_annotations(image_dir: str, output_file: str, 
                               default_label: int = 0) -> None:
    """
    Create sample annotations for all images in a directory.
    
    All cells are initialized with the default label.
    This provides a template that can be manually edited.
    
    Args:
        image_dir: Directory containing images
        output_file: Path to save the annotations
        default_label: Default label for all cells (default 0 = no_object)
    """
    from .utils import get_image_files
    
    image_files = get_image_files(image_dir)
    annotations = []
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        ann = create_annotation(filename, {})
        
        # Set all cells to default label
        ann['labels'] = [default_label] * TOTAL_CELLS
        
        annotations.append(ann)
    
    save_annotations_batch(annotations, output_file)
    print(f"Created sample annotations for {len(annotations)} images")


def validate_annotation(annotation: Dict) -> bool:
    """
    Validate an annotation dictionary.
    
    Args:
        annotation: Annotation dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    required_fields = ['image_filename', 'labels']
    for field in required_fields:
        if field not in annotation:
            print(f"Missing required field: {field}")
            return False
    
    # Check labels
    labels = annotation['labels']
    if len(labels) != TOTAL_CELLS:
        print(f"Expected {TOTAL_CELLS} labels, got {len(labels)}")
        return False
    
    for i, label in enumerate(labels):
        if label not in [0, 1, 2, 3]:
            print(f"Invalid label {label} at position {i}")
            return False
    
    return True


def interactive_annotate(image_path: str, output_path: str = None) -> Dict:
    """
    Interactive annotation helper that prints grid reference.
    
    This is a simple text-based helper for annotation.
    For GUI annotation, consider using external tools.
    
    Args:
        image_path: Path to the image
        output_path: Optional path to save the annotation
    
    Returns:
        Annotation dictionary
    """
    image_filename = os.path.basename(image_path)
    
    print(f"\n{'='*60}")
    print(f"Annotating: {image_filename}")
    print(f"{'='*60}")
    print(f"\nGrid layout (8x8 = 64 cells):")
    print(f"Cell indices are numbered c01-c64 (row-major order)")
    print(f"\n  ", end="")
    for col in range(1, GRID_COLS + 1):
        print(f"  Col{col}", end="")
    print()
    
    for row in range(GRID_ROWS):
        print(f"Row{row+1}: ", end="")
        for col in range(GRID_COLS):
            cell_idx = row * GRID_COLS + col + 1
            print(f"  c{cell_idx:02d} ", end="")
        print()
    
    print(f"\nLabels: 0=no_object, 1=ball, 2=bat, 3=stump")
    print(f"\nEnter labels in format: cell_index:label (e.g., 35:1 for ball in c35)")
    print(f"Separate multiple entries with spaces.")
    print(f"Press Enter when done (unlabeled cells default to 0).")
    
    user_input = input("\nLabels: ").strip()
    
    cell_labels = {}
    if user_input:
        entries = user_input.split()
        for entry in entries:
            try:
                cell_str, label_str = entry.split(':')
                cell_idx = int(cell_str.replace('c', ''))
                label = int(label_str)
                cell_labels[cell_idx] = label
            except (ValueError, IndexError):
                print(f"Warning: Could not parse '{entry}', skipping...")
    
    annotation = create_annotation(image_filename, cell_labels)
    
    if output_path:
        save_annotation(annotation, output_path)
    
    return annotation
