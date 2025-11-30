# Cricket Image Classification Project

## Team Name: Team_Pravin

## Project Description
This project implements a grid-based cricket object detection model that classifies regions of cricket images into four categories:
- **0**: No object
- **1**: Ball
- **2**: Bat
- **3**: Stump

## Dataset Description

### Image Requirements
- **Aspect Ratio**: 4:3
- **Resolution**: 800 x 600 pixels
- **Minimum Dataset Size**: 300 images

### Image Categories
1. **Ball Images**: Cricket balls in various positions (on ground, in air, in hand)
2. **Bat Images**: Cricket bats from various angles (partial/complete views)
3. **Stump Images**: Wicket stumps (clear, occluded, etc.)
4. **No-Object Images**: Background scenes (grass, pitch, stadium)

### Image Sources
Images for this dataset can be collected from:
1. Google Images (search for cricket-related images)
2. Getty Images (cricket stock photos)
3. Unsplash (free cricket photos)
4. Pexels (free cricket images)
5. Cricket-specific websites
6. Personal cricket match photographs

**Note**: Always respect copyright and licensing terms when using images.

## Directory Structure
```
CricketImageClassification/
├── data/
│   ├── train/
│   │   ├── ball/
│   │   ├── bat/
│   │   ├── stump/
│   │   └── no_object/
│   ├── test/
│   │   ├── ball/
│   │   ├── bat/
│   │   ├── stump/
│   │   └── no_object/
│   └── annotations/
│       └── annotations.json
├── models/
│   └── model_Team_Pravin.pkl
├── outputs/
│   └── predictions.csv
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── model.py
│   └── utils.py
├── main.py
└── README.md
```

## How to Use

### 1. Prepare Dataset
Place images in the appropriate data folders (train/test, categorized by object type).

### 2. Create Annotations
Run the annotation tool to label grid cells:
```bash
python main.py --mode annotate
```

### 3. Train Model
```bash
python main.py --mode train
```

### 4. Generate Predictions
```bash
python main.py --mode predict
```

## Feature Engineering
This project uses hand-crafted features (no CNNs):
- Histogram of Oriented Gradients (HOG)
- Color Histograms
- Edge Detection Features
- Texture Features (LBP)
- Shape Features

## Output Format
The prediction CSV has the following format:
```
ImageFileName, TrainOrTest, c01, c02, ..., c64
```
Where each cell value is 0, 1, 2, or 3.
