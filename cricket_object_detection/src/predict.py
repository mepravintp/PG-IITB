"""
Prediction module for cricket object detection.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image

from .utils import (
    TOTAL_CELLS, CLASS_LABELS, load_model, get_image_files,
    create_csv_header, create_csv_row
)
from .preprocess import preprocess_image, image_to_numpy
from .features import extract_grid_features


def load_trained_model(model_path: str) -> Tuple:
    """
    Load a trained model with preprocessing parameters.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Tuple of (model, mean, std)
    """
    model_data = load_model(model_path)
    
    model = model_data['model']
    mean = model_data['mean']
    std = model_data['std']
    
    print(f"Loaded model: {model_data.get('team_name', 'unknown')}")
    
    return model, mean, std


def predict_image(image_path: str, model, mean: np.ndarray, 
                   std: np.ndarray) -> List[int]:
    """
    Predict grid cell labels for a single image.
    
    Args:
        image_path: Path to the image
        model: Trained classifier
        mean: Feature mean for standardization
        std: Feature std for standardization
    
    Returns:
        List of 64 predictions (0-3 for each cell)
    """
    # Preprocess image
    processed = preprocess_image(image_path)
    
    if processed is None:
        print(f"Warning: Could not process image: {image_path}")
        return [0] * TOTAL_CELLS
    
    # Convert to numpy array
    image_array = image_to_numpy(processed)
    
    # Extract features for all cells
    cell_features = extract_grid_features(image_array)
    
    # Stack features into matrix
    X = np.array(cell_features)
    
    # Standardize
    X_scaled = (X - mean) / std
    
    # Predict
    predictions = model.predict(X_scaled)
    
    return predictions.tolist()


def predict_batch(image_paths: List[str], model, mean: np.ndarray,
                   std: np.ndarray) -> Dict[str, List[int]]:
    """
    Predict grid cell labels for multiple images.
    
    Args:
        image_paths: List of image paths
        model: Trained classifier
        mean: Feature mean for standardization
        std: Feature std for standardization
    
    Returns:
        Dictionary mapping image filenames to prediction lists
    """
    results = {}
    
    for idx, image_path in enumerate(image_paths):
        if (idx + 1) % 10 == 0:
            print(f"Predicting image {idx + 1}/{len(image_paths)}...")
        
        filename = os.path.basename(image_path)
        predictions = predict_image(image_path, model, mean, std)
        results[filename] = predictions
    
    return results


def predict_directory(image_dir: str, model_path: str) -> Dict[str, List[int]]:
    """
    Predict grid cell labels for all images in a directory.
    
    Args:
        image_dir: Directory containing images
        model_path: Path to the trained model
    
    Returns:
        Dictionary mapping image filenames to prediction lists
    """
    # Load model
    model, mean, std = load_trained_model(model_path)
    
    # Get image files
    image_paths = get_image_files(image_dir)
    
    if len(image_paths) == 0:
        print(f"No images found in {image_dir}")
        return {}
    
    print(f"Found {len(image_paths)} images to predict")
    
    # Predict
    results = predict_batch(image_paths, model, mean, std)
    
    return results


def generate_csv_output(train_predictions: Dict[str, List[int]],
                         test_predictions: Dict[str, List[int]],
                         output_path: str) -> None:
    """
    Generate the final CSV output file.
    
    Args:
        train_predictions: Dictionary of train image predictions
        test_predictions: Dictionary of test image predictions
        output_path: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write header
        f.write(create_csv_header() + '\n')
        
        # Write train predictions
        for filename, predictions in sorted(train_predictions.items()):
            row = create_csv_row(filename, 'Train', predictions)
            f.write(row + '\n')
        
        # Write test predictions
        for filename, predictions in sorted(test_predictions.items()):
            row = create_csv_row(filename, 'Test', predictions)
            f.write(row + '\n')
    
    total_rows = len(train_predictions) + len(test_predictions)
    print(f"CSV output saved to {output_path} ({total_rows} rows)")


def run_predictions(train_dir: str, test_dir: str, model_path: str,
                     output_path: str) -> None:
    """
    Run predictions on train and test datasets and save CSV output.
    
    Args:
        train_dir: Directory containing training images
        test_dir: Directory containing test images
        model_path: Path to the trained model
        output_path: Path to save the CSV output
    """
    # Load model
    model, mean, std = load_trained_model(model_path)
    
    # Predict train set
    print("\n=== Predicting on training set ===")
    train_paths = get_image_files(train_dir)
    train_predictions = predict_batch(train_paths, model, mean, std)
    
    # Predict test set
    print("\n=== Predicting on test set ===")
    test_paths = get_image_files(test_dir)
    test_predictions = predict_batch(test_paths, model, mean, std)
    
    # Generate CSV
    generate_csv_output(train_predictions, test_predictions, output_path)


def evaluate_predictions(predictions: Dict[str, List[int]],
                          annotations_file: str) -> Dict:
    """
    Evaluate predictions against ground truth annotations.
    
    Args:
        predictions: Dictionary of predictions
        annotations_file: Path to annotations JSON file
    
    Returns:
        Dictionary with evaluation metrics
    """
    from .annotate import load_annotations_batch
    
    annotations = load_annotations_batch(annotations_file)
    ann_dict = {ann['image_filename']: ann['labels'] for ann in annotations}
    
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(4)}
    class_total = {i: 0 for i in range(4)}
    
    for filename, pred_labels in predictions.items():
        if filename not in ann_dict:
            continue
        
        true_labels = ann_dict[filename]
        
        for pred, true in zip(pred_labels, true_labels):
            if pred == true:
                correct += 1
                class_correct[true] += 1
            
            class_total[true] += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    class_accuracy = {}
    for cls in range(4):
        if class_total[cls] > 0:
            class_accuracy[CLASS_LABELS[cls]] = class_correct[cls] / class_total[cls]
        else:
            class_accuracy[CLASS_LABELS[cls]] = 0
    
    return {
        'overall_accuracy': accuracy,
        'total_cells': total,
        'correct_predictions': correct,
        'class_accuracy': class_accuracy,
        'class_counts': {CLASS_LABELS[k]: v for k, v in class_total.items()}
    }


def print_evaluation_report(metrics: Dict) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary with evaluation metrics
    """
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Total Cells Evaluated: {metrics['total_cells']}")
    print(f"Correct Predictions: {metrics['correct_predictions']}")
    print("\nPer-Class Accuracy:")
    for cls, acc in metrics['class_accuracy'].items():
        count = metrics['class_counts'][cls]
        print(f"  {cls}: {acc:.4f} ({count} samples)")
    print("=" * 50)


def visualize_predictions(image_path: str, predictions: List[int],
                           output_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize predictions on an image.
    
    Args:
        image_path: Path to the image
        predictions: List of 64 predictions
        output_path: Optional path to save the visualization
    
    Returns:
        Annotated image array
    """
    from .annotate import visualize_annotation
    
    processed = preprocess_image(image_path)
    
    if processed is None:
        raise ValueError(f"Could not process image: {image_path}")
    
    image_array = image_to_numpy(processed)
    
    return visualize_annotation(image_array, predictions, output_path)


def demo_prediction(image_path: str, model_path: str) -> List[int]:
    """
    Demo function to predict and display results for a single image.
    
    Args:
        image_path: Path to the image
        model_path: Path to the trained model
    
    Returns:
        List of 64 predictions
    """
    # Load model
    model, mean, std = load_trained_model(model_path)
    
    # Predict
    predictions = predict_image(image_path, model, mean, std)
    
    # Print results
    print(f"\nPredictions for: {os.path.basename(image_path)}")
    print("-" * 40)
    
    # Print grid
    for row in range(8):
        row_preds = predictions[row*8:(row+1)*8]
        print(f"Row {row+1}: ", end="")
        for pred in row_preds:
            label = CLASS_LABELS[pred][:4]
            print(f"{label:5s}", end=" ")
        print()
    
    # Count objects
    counts = {}
    for pred in predictions:
        label = CLASS_LABELS[pred]
        counts[label] = counts.get(label, 0) + 1
    
    print("\nObject counts:")
    for label, count in counts.items():
        if label != 'no_object':
            print(f"  {label}: {count}")
    
    return predictions
