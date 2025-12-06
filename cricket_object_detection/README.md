# Cricket Object Detection Project

## Overview
This project implements a grid-based object detection system for cricket images. It detects cricket bats, balls, and stumps in images using hand-crafted features and traditional machine learning classifiers.

## Project Task List
See [TASKS.csv](TASKS.csv) for the editable Excel-like team task table.

## Project Structure
```
cricket_object_detection/
├── data/
│   ├── raw/                # Original images (unprocessed)
│   │   ├── bat/
│   │   ├── ball/
│   │   ├── stumps/
│   │   └── no_object/
│   ├── train/              # Preprocessed training images (by class)
│   │   ├── bat/
│   │   ├── ball/
│   │   ├── stumps/
│   │   └── no_object/
│   ├── test/               # Preprocessed test images (by class)
│   ├── annotations/        # Annotation files (JSON/CSV)
│   └── processed_images/   # Images with grid/labels overlay
│
├── models/                 # Saved model files
├── outputs/                # Prediction CSVs, evaluation reports, visualizations
├── notebooks/              # Jupyter notebooks for EDA, training, analysis
├── src/                    # Source code (preprocessing, training, annotation, etc.)
│   ├── preprocess.py
│   ├── annotate.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── scripts/                # Utility scripts (manual annotation, batch processing)
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and instructions
└── TASKS.md / TASKS.csv    # Team task list and assignments
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
Team Name: TBD

## Requirements
- Python 3.7+
- NumPy
- scikit-learn
- scikit-image
- OpenCV-Python
- Pillow

# Cricket Object Detection Project - README

## Folder Structure

```
cricket_object_detection/
│
├── data/
│   ├── raw/                # Original images (unprocessed)
│   │   ├── bat/
│   │   ├── ball/
│   │   ├── stumps/
│   │   └── no_object/
│   ├── train/              # Preprocessed training images (by class)
│   │   ├── bat/
│   │   ├── ball/
│   │   ├── stumps/
│   │   └── no_object/
│   ├── test/               # Preprocessed test images (by class)
│   ├── annotations/        # Annotation files (JSON/CSV)
│   └── processed_images/   # Images with grid/labels overlay
│
├── models/                 # Saved model files
├── outputs/                # Prediction CSVs, evaluation reports, visualizations
├── notebooks/              # Jupyter notebooks for EDA, training, analysis
├── src/                    # Source code (preprocessing, training, annotation, etc.)
│   ├── preprocess.py
│   ├── annotate.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── scripts/                # Utility scripts (manual annotation, batch processing)
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and instructions
└── TASKS.md / TASKS.csv    # Team task list and assignments
```

## Step-by-Step Tasks

1. **Data Collection**
   - Gather cricket images for each class: bat, ball, stumps, no_object.
   - Place raw images in `data/raw/<class>/`.

2. **Preprocessing**
   - Resize/crop images to 800x600 pixels (4:3 aspect ratio).
   - Save processed images in `data/train/<class>/` and `data/test/<class>/`.

3. **Annotation**
   - Annotate each image with grid cell labels (0: no object, 1: ball, 2: bat, 3: stump).
   - Store annotation files in `data/annotations/` (e.g., `train_annotations.json`).

4. **Manual Annotation (Optional)**
   - Use the provided GUI tool or script to tag grid cells interactively.
   - Save results to CSV or JSON in `data/annotations/`.

5. **Feature Extraction**
   - Extract hand-crafted features from each grid cell using scripts in `src/`.

6. **Model Training**
   - Train a classifier (e.g., Random Forest) using extracted features and annotations.
   - Save trained models in `models/`.

7. **Prediction**
   - Run predictions on test images.
   - Save per-cell predictions to CSV in `outputs/`.

8. **Evaluation**
   - Compare predictions with ground truth annotations.
   - Generate evaluation reports in `outputs/`.

9. **Visualization**
   - Overlay grid and predicted/true labels on images for review.
   - Save visualizations in `data/processed_images/` or `outputs/`.

10. **Collaboration & Task Tracking**
    - Assign and track tasks in `TASKS.md` or `TASKS.csv`.

---

**For details on each step, see the relevant scripts and notebooks in the project.**
