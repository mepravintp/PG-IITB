"""
Cricket Object Detection Package

This package implements a grid-based object detection system for cricket images
using hand-crafted features and traditional machine learning classifiers.

Modules:
    - utils: Common utility functions
    - preprocess: Image preprocessing utilities
    - features: Hand-crafted feature extraction
    - annotate: Annotation utilities
    - train: Model training pipeline
    - predict: Prediction module
"""

from .utils import (
    CLASS_LABELS, LABEL_TO_CLASS,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    GRID_ROWS, GRID_COLS,
    CELL_WIDTH, CELL_HEIGHT,
    TOTAL_CELLS
)

__version__ = '1.0.0'
__author__ = 'PravinTeam'
