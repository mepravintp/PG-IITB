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

1. **Data Collection** - done
   - Gather cricket images for each class: bat, ball, stumps, no_object.
   - Place raw images in `data/raw/<class>/`.

2. **Preprocessing** - Ratish will create
   - Resize/crop images to 800x600 pixels (4:3 aspect ratio).
   - Save processed images in `data/train/<class>/` and `data/test/<class>/`.

3. **Annotation** 
   - Annotate each image with grid cell labels (0: no object, 1: ball, 2: bat, 3: stump).
   - Store annotation files in `data/annotations/` (e.g., `train_annotations.json`).

   YOgesh - Balls
   Yugal- Bats
   Ratish - Stumps

  




4. **Manual Annotation (Optional)**
   - Use the provided GUI tool or script to tag grid cells interactively.
   - Save results to CSV or JSON in `data/annotations/`.

    Task :
      clone develop branch
      pre process image by using preprocess.py .by this we will have preprocessed images in train/$label.
      run annotate utillity and tag bat,ball stumps for your respective images.
      

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
