#!/usr/bin/env python3
"""
Cricket Image Classification - Main Script
This script provides the main pipeline for training and predicting
cricket object detection in grid cells.

Usage:
    python main.py --mode train
    python main.py --mode predict
    python main.py --mode annotate
    python main.py --mode evaluate
"""

import argparse
import os
import sys
import numpy as np
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import (
    load_and_preprocess_image, 
    divide_into_grid, 
    visualize_grid
)
from src.features import extract_cell_features, extract_features_from_grid
from src.model import (
    GridClassifier, 
    RandomForestSimple, 
    evaluate_model, 
    LABEL_NAMES
)
from src.utils import (
    save_predictions_to_csv, 
    load_annotations, 
    save_annotations,
    create_sample_annotations,
    summarize_predictions,
    validate_dataset_structure,
    count_images,
    print_confusion_matrix
)


# Configuration
TEAM_NAME = "Team_Pravin"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
ANNOTATIONS_FILE = os.path.join(DATA_DIR, 'annotations', 'annotations.json')


def load_dataset(data_dir: str, annotations: Dict[str, List[int]], 
                 data_type: str = 'train') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and process dataset for training/testing.
    
    Args:
        data_dir: Base data directory
        annotations: Dictionary of cell annotations
        data_type: 'train' or 'test'
        
    Returns:
        Tuple of (features, labels, filenames)
    """
    all_features = []
    all_labels = []
    all_filenames = []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    categories = ['ball', 'bat', 'stump', 'no_object']
    
    for category in categories:
        category_dir = os.path.join(data_dir, data_type, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Directory not found: {category_dir}")
            continue
        
        for filename in os.listdir(category_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                continue
            
            image_path = os.path.join(category_dir, filename)
            image = load_and_preprocess_image(image_path)
            
            if image is None:
                continue
            
            # Divide into grid and extract features
            cells = divide_into_grid(image)
            cell_features = extract_features_from_grid(cells)
            
            # Get annotations for this image
            full_filename = f"{data_type}/{category}/{filename}"
            if full_filename in annotations:
                cell_labels = annotations[full_filename]
            else:
                # Default: use category label for all cells
                # In reality, you'd want proper per-cell annotations
                category_label = {'no_object': 0, 'ball': 1, 'bat': 2, 'stump': 3}[category]
                cell_labels = [category_label] * 64
            
            all_features.append(cell_features)
            all_labels.extend(cell_labels)
            all_filenames.append(full_filename)
    
    if len(all_features) == 0:
        return np.array([]), np.array([]), []
    
    # Stack features from all images
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    return X, y, all_filenames


def train_model(use_random_forest: bool = False, verbose: bool = True) -> None:
    """
    Train the grid classification model.
    
    Args:
        use_random_forest: Whether to use Random Forest instead of Softmax Regression
        verbose: Print training progress
    """
    print("=" * 60)
    print("CRICKET IMAGE CLASSIFICATION - TRAINING")
    print("=" * 60)
    
    # Validate dataset structure
    if not validate_dataset_structure(BASE_DIR):
        print("\nDataset structure is incomplete. Please create missing directories.")
        return
    
    # Print dataset statistics
    print("\nDataset statistics:")
    counts = count_images(BASE_DIR)
    total = 0
    for key, count in counts.items():
        print(f"  {key}: {count} images")
        total += count
    print(f"  Total: {total} images")
    
    if total == 0:
        print("\nNo images found! Please add images to the data directory.")
        print("Creating sample annotation template for when images are added...")
        return
    
    # Load annotations
    if os.path.exists(ANNOTATIONS_FILE):
        annotations = load_annotations(ANNOTATIONS_FILE)
        print(f"\nLoaded annotations for {len(annotations)} images")
    else:
        print("\nNo annotations found. Using default category-based labeling.")
        annotations = {}
    
    # Load training data
    print("\nLoading training data...")
    X_train, y_train, train_files = load_dataset(DATA_DIR, annotations, 'train')
    
    if len(X_train) == 0:
        print("No training data found!")
        return
    
    print(f"Training samples: {X_train.shape[0]} cells from {len(train_files)} images")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Class distribution
    print("\nClass distribution in training data:")
    for class_id in range(4):
        count = np.sum(y_train == class_id)
        print(f"  {LABEL_NAMES[class_id]}: {count} ({count/len(y_train)*100:.2f}%)")
    
    # Train model
    print("\nTraining classifier...")
    if use_random_forest:
        model = RandomForestSimple(n_trees=50, max_depth=10)
    else:
        model = GridClassifier(learning_rate=0.1, n_iterations=1000, regularization=0.01)
    
    model.fit(X_train, y_train, verbose=verbose)
    
    # Evaluate on training data
    print("\nEvaluating on training data...")
    y_pred_train = model.predict(X_train)
    metrics = evaluate_model(y_train, y_pred_train)
    
    print("\nTraining metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f'model_{TEAM_NAME}.pkl')
    model.save(model_path)
    
    print("\nTraining complete!")


def predict(model_path: str = None, verbose: bool = True) -> None:
    """
    Generate predictions for all images.
    
    Args:
        model_path: Path to the trained model
        verbose: Print progress
    """
    print("=" * 60)
    print("CRICKET IMAGE CLASSIFICATION - PREDICTION")
    print("=" * 60)
    
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, f'model_{TEAM_NAME}.pkl')
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first using: python main.py --mode train")
        return
    
    # Load model
    print(f"\nLoading model from {model_path}")
    model = GridClassifier.load(model_path)
    
    # Prepare predictions dictionary
    predictions = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for data_type in ['train', 'test']:
        for category in ['ball', 'bat', 'stump', 'no_object']:
            category_dir = os.path.join(DATA_DIR, data_type, category)
            if not os.path.exists(category_dir):
                continue
            
            for filename in os.listdir(category_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in valid_extensions:
                    continue
                
                image_path = os.path.join(category_dir, filename)
                image = load_and_preprocess_image(image_path)
                
                if image is None:
                    continue
                
                # Divide into grid and extract features
                cells = divide_into_grid(image)
                cell_features = extract_features_from_grid(cells)
                
                # Predict
                cell_predictions = model.predict(cell_features)
                
                # Store prediction
                full_filename = f"{data_type}/{category}/{filename}"
                predictions[full_filename] = (data_type, cell_predictions.tolist())
                
                if verbose:
                    print(f"Processed: {full_filename}")
    
    # Save predictions to CSV
    output_path = os.path.join(OUTPUTS_DIR, 'predictions.csv')
    save_predictions_to_csv(predictions, output_path)
    
    # Print summary
    summarize_predictions(predictions)
    
    print("\nPrediction complete!")


def evaluate(model_path: str = None) -> None:
    """
    Evaluate the model on test data.
    
    Args:
        model_path: Path to the trained model
    """
    print("=" * 60)
    print("CRICKET IMAGE CLASSIFICATION - EVALUATION")
    print("=" * 60)
    
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, f'model_{TEAM_NAME}.pkl')
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Load model
    model = GridClassifier.load(model_path)
    
    # Load annotations
    if os.path.exists(ANNOTATIONS_FILE):
        annotations = load_annotations(ANNOTATIONS_FILE)
    else:
        annotations = {}
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test, test_files = load_dataset(DATA_DIR, annotations, 'test')
    
    if len(X_test) == 0:
        print("No test data found!")
        return
    
    print(f"Test samples: {X_test.shape[0]} cells from {len(test_files)} images")
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Print confusion matrix
    print_confusion_matrix(y_test, y_pred)


def create_annotation_template() -> None:
    """
    Create annotation templates for all images.
    """
    print("=" * 60)
    print("CREATING ANNOTATION TEMPLATE")
    print("=" * 60)
    
    annotations = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for data_type in ['train', 'test']:
        for category in ['ball', 'bat', 'stump', 'no_object']:
            category_dir = os.path.join(DATA_DIR, data_type, category)
            if not os.path.exists(category_dir):
                continue
            
            for filename in os.listdir(category_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:
                    full_filename = f"{data_type}/{category}/{filename}"
                    # Default: all cells as no_object
                    annotations[full_filename] = [0] * 64
    
    # Save annotations template
    os.makedirs(os.path.dirname(ANNOTATIONS_FILE), exist_ok=True)
    save_annotations(annotations, ANNOTATIONS_FILE)
    
    print(f"\nCreated annotation template for {len(annotations)} images")
    print(f"Template saved to: {ANNOTATIONS_FILE}")
    print("\nPlease manually edit this file to add cell-level annotations.")
    print("Each cell value should be:")
    print("  0 - No object")
    print("  1 - Ball")
    print("  2 - Bat")
    print("  3 - Stump")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Cricket Image Classification Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'predict', 'evaluate', 'annotate'],
        required=True,
        help='Mode of operation: train, predict, evaluate, or annotate'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default=None,
        help='Path to model file (for predict/evaluate modes)'
    )
    
    parser.add_argument(
        '--random-forest',
        action='store_true',
        help='Use Random Forest instead of Softmax Regression'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(use_random_forest=args.random_forest, verbose=args.verbose)
    elif args.mode == 'predict':
        predict(model_path=args.model, verbose=args.verbose)
    elif args.mode == 'evaluate':
        evaluate(model_path=args.model)
    elif args.mode == 'annotate':
        create_annotation_template()


if __name__ == '__main__':
    main()
