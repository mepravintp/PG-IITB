#!/usr/bin/env python3
"""
Main script for the Cricket Object Detection project.

This script provides command-line interface for:
    - Training the model
    - Making predictions
    - Evaluating results
    - Generating CSV output
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import TOTAL_CELLS, CLASS_LABELS
from src.preprocess import preprocess_directory, preprocess_image
from src.annotate import (
    create_annotation, save_annotations_batch, 
    load_annotations_batch, create_sample_annotations
)
from src.train import train_from_directory, cross_validate, prepare_training_data, load_training_data
from src.predict import (
    run_predictions, predict_directory, 
    evaluate_predictions, print_evaluation_report,
    load_trained_model
)


def cmd_preprocess(args):
    """Preprocess images in a directory."""
    print(f"Preprocessing images from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    processed, skipped = preprocess_directory(args.input_dir, args.output_dir)
    
    print(f"\nProcessed: {processed} images")
    print(f"Skipped: {skipped} images")


def cmd_annotate(args):
    """Create sample annotations for images."""
    print(f"Creating sample annotations for: {args.image_dir}")
    print(f"Output file: {args.output_file}")
    
    create_sample_annotations(args.image_dir, args.output_file)


def cmd_train(args):
    """Train the model."""
    print(f"Training model with images from: {args.image_dir}")
    print(f"Annotations file: {args.annotations}")
    print(f"Team name: {args.team_name}")
    print(f"Model type: {args.model_type}")
    
    model_path = train_from_directory(
        train_image_dir=args.image_dir,
        annotations_file=args.annotations,
        team_name=args.team_name,
        output_dir=args.output_dir,
        model_type=args.model_type
    )
    
    print(f"\nModel saved to: {model_path}")


def cmd_predict(args):
    """Run predictions and generate CSV output."""
    print(f"Running predictions...")
    print(f"Train directory: {args.train_dir}")
    print(f"Test directory: {args.test_dir}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    
    run_predictions(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_path=args.model,
        output_path=args.output
    )


def cmd_evaluate(args):
    """Evaluate predictions against ground truth."""
    print(f"Evaluating predictions...")
    print(f"Image directory: {args.image_dir}")
    print(f"Model: {args.model}")
    print(f"Annotations: {args.annotations}")
    
    # Make predictions
    predictions = predict_directory(args.image_dir, args.model)
    
    # Evaluate
    metrics = evaluate_predictions(predictions, args.annotations)
    
    # Print report
    print_evaluation_report(metrics)


def cmd_demo(args):
    """Demo the complete pipeline with sample data."""
    print("=" * 60)
    print("CRICKET OBJECT DETECTION - DEMO")
    print("=" * 60)
    
    print("""
This demo shows how to use the cricket object detection system.

Required steps:
1. Collect cricket images (minimum 300)
2. Preprocess images to 800x600 with 4:3 aspect ratio
3. Create annotations for each image (64 grid cells per image)
4. Train the model
5. Run predictions on train and test sets
6. Generate CSV output

Classes:
    0 - No object
    1 - Ball
    2 - Bat
    3 - Stump

Grid Layout:
    Each 800x600 image is divided into 8x8 = 64 cells
    Each cell is 100x75 pixels
    Cells are numbered c01 to c64 (row-major order)

Usage Example:
    # Preprocess images
    python main.py preprocess --input-dir raw_images --output-dir data/train/bat
    
    # Create sample annotations (to be edited manually)
    python main.py annotate --image-dir data/train --output-file annotations/train.json
    
    # Train model
    python main.py train --image-dir data/train --annotations annotations/train.json \\
                         --team-name MyTeam --model-type random_forest
    
    # Run predictions
    python main.py predict --train-dir data/train --test-dir data/test \\
                          --model models/model_MyTeam.pkl --output outputs/predictions.csv
    
    # Evaluate
    python main.py evaluate --image-dir data/train --model models/model_MyTeam.pkl \\
                           --annotations annotations/train.json
""")


def main():
    parser = argparse.ArgumentParser(
        description='Cricket Object Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        'preprocess', 
        help='Preprocess images (resize to 800x600, crop to 4:3)'
    )
    preprocess_parser.add_argument('--input-dir', required=True, 
                                    help='Input directory with raw images')
    preprocess_parser.add_argument('--output-dir', required=True,
                                    help='Output directory for processed images')
    preprocess_parser.set_defaults(func=cmd_preprocess)
    
    # Annotate command
    annotate_parser = subparsers.add_parser(
        'annotate',
        help='Create sample annotations for images'
    )
    annotate_parser.add_argument('--image-dir', required=True,
                                  help='Directory containing images')
    annotate_parser.add_argument('--output-file', required=True,
                                  help='Output JSON file for annotations')
    annotate_parser.set_defaults(func=cmd_annotate)
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train the object detection model'
    )
    train_parser.add_argument('--image-dir', required=True,
                               help='Directory containing training images')
    train_parser.add_argument('--annotations', required=True,
                               help='Path to annotations JSON file')
    train_parser.add_argument('--team-name', default='PravinTeam',
                               help='Team name for model file')
    train_parser.add_argument('--output-dir', default='models',
                               help='Directory to save the trained model')
    train_parser.add_argument('--model-type', default='random_forest',
                               choices=['logistic', 'random_forest'],
                               help='Type of classifier to use')
    train_parser.set_defaults(func=cmd_train)
    
    # Predict command
    predict_parser = subparsers.add_parser(
        'predict',
        help='Run predictions and generate CSV output'
    )
    predict_parser.add_argument('--train-dir', required=True,
                                 help='Directory containing training images')
    predict_parser.add_argument('--test-dir', required=True,
                                 help='Directory containing test images')
    predict_parser.add_argument('--model', required=True,
                                 help='Path to trained model file')
    predict_parser.add_argument('--output', required=True,
                                 help='Path for output CSV file')
    predict_parser.set_defaults(func=cmd_predict)
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate predictions against ground truth'
    )
    evaluate_parser.add_argument('--image-dir', required=True,
                                  help='Directory containing images')
    evaluate_parser.add_argument('--model', required=True,
                                  help='Path to trained model file')
    evaluate_parser.add_argument('--annotations', required=True,
                                  help='Path to ground truth annotations')
    evaluate_parser.set_defaults(func=cmd_evaluate)
    
    # Demo command
    demo_parser = subparsers.add_parser(
        'demo',
        help='Show usage examples and documentation'
    )
    demo_parser.set_defaults(func=cmd_demo)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()
