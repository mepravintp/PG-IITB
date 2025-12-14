# Code Directory - File Explanations

This README provides a brief 2-line explanation for each file in this directory.

---

**part1_preprocess.py**  
Image preprocessing utilities for cricket object detection.  
Includes functions for validating and resizing images before annotation or model training.  
Handles image size checks, normalization, and prepares images for consistent downstream processing.  
Essential for ensuring data quality before annotation and model input.

**part2_annotations.ipynb**  
Interactive notebook for annotating cricket images with an 8x8 grid.  
Provides a standalone interface to label images for object detection tasks.  
Allows users to select, view, and annotate images interactively.  
Annotations are saved for use in training and evaluation.

**part3_annotation_to_csv.py**  
Script to convert image annotation text files into a consolidated CSV.  
Automates the process of gathering annotation data for further analysis or training.  
Scans directories for annotation files and matches them with images.  
Outputs a CSV file suitable for machine learning pipelines.

**part4_check_data_imbalance.ipynb**  
Notebook to analyze and visualize class/data imbalance in the dataset.  
Helps identify if certain classes or labels are underrepresented.  
Loads annotation data and computes class distributions.  
Provides visualizations to guide data collection or augmentation.

**part5_image_classificaton.ipynb**  
Notebook for building and evaluating an image classification pipeline.  
Covers data loading, feature extraction, model training, and evaluation steps.  
Implements cell-level feature extraction and model selection.  
Includes code for training, validation, and performance visualization.

**part6_visualize_predictions.ipynb**  
Notebook to visualize model predictions by overlaying results on images.  
Reads prediction CSVs and saves images with color-coded cell grids for inspection.  
Helps in qualitative assessment of model performance.  
Useful for debugging and presenting results visually.
