# Cricket Object Detection Project

## Overview
This project implements a grid-based object detection system for cricket images. It detects cricket bats, balls, and stumps in images using hand-crafted features and traditional machine learning classifiers.

## Project Structure
```
cricket_object_detection/
├── data/
│   ├── train/
│   │   ├── bat/          # Training images containing bats
│   │   ├── ball/         # Training images containing balls
│   │   ├── stumps/       # Training images containing stumps
│   │   └── no_object/    # Training images with no cricket objects
│   └── test/
│       ├── bat/          # Test images containing bats
│       ├── ball/         # Test images containing balls
│       ├── stumps/       # Test images containing stumps
│       └── no_object/    # Test images with no cricket objects
├── annotations/          # Grid-level annotations (JSON format)
├── models/               # Saved trained models
├── outputs/              # CSV output files
├── src/
│   ├── preprocess.py     # Image preprocessing utilities
│   ├── features.py       # Hand-crafted feature extraction
│   ├── annotate.py       # Annotation utilities
│   ├── train.py          # Model training pipeline
│   ├── predict.py        # Prediction module
│   └── utils.py          # Common utility functions
└── README.md
```

## Dataset Requirements
- Minimum 300 images (balanced across categories)
- All images must have 4:3 aspect ratio
- Images resized to 800 x 600 pixels
- Images with resolution lower than 800x600 should NOT be scaled up/used

## Image Sources
The dataset images should be collected from:
- Cricket match screenshots and photographs
- Sports photography websites (with proper attribution)
- User-captured cricket photographs
- Public domain cricket image repositories

**Note**: All images should be free to use or properly licensed.

## Grid-Based Detection
Each 800x600 image is divided into an 8x8 grid (64 cells):
- Each cell is 100x75 pixels
- Grid cell numbering: c01 to c64 (row-major order)

## Class Labels
- 0 → No object
- 1 → Ball
- 2 → Bat
- 3 → Stump

## Usage

### 1. Image Preprocessing
```python
from src.preprocess import preprocess_image, validate_image_size
```

### 2. Annotation
```python
from src.annotate import create_annotation, load_annotations
```

### 3. Feature Extraction
```python
from src.features import extract_grid_features
```

### 4. Training
```python
from src.train import train_model
```

### 5. Prediction
```python
from src.predict import predict_image, generate_csv_output
```

## Model Output
The trained model is saved as `model_<teamname>.pkl` in the models folder.

## CSV Output Format
```
ImageFileName, TrainOrTest, c01, c02, ..., c64
```

Where each cell value is 0, 1, 2, or 3 indicating the detected object class.

## Team
Team Name: PravinTeam

## Requirements
- Python 3.7+
- NumPy
- scikit-learn
- scikit-image
- OpenCV-Python
- Pillow
