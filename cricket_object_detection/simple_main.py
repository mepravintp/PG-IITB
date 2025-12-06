import sys
import os
from src.preprocess import preprocess_directory
from src.annotate import create_sample_annotations, visualize_annotation
import numpy as np
from PIL import Image
import json

def main():
    
    print("Current working directory:", os.getcwd())
    input_dir = 'data/input'  # Change as needed
    output_dir = 'data/train/bat'  # Change as needed
    
    print(f"Preprocessing images from: {input_dir}")
    #list all files from input_dir
    files = os.listdir(input_dir)
    print(f"Files found: {files}")
    
    print(f"Output directory: {output_dir}")
    processed, skipped = preprocess_directory(input_dir, output_dir)
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")

    # After preprocessing, create sample annotations for the same files
    annotation_output = 'data/train/bat/sample_annotations.json'  # Change as needed
    create_sample_annotations(input_dir, annotation_output)
    print(f"Sample annotations created at: {annotation_output}")

    # Visualize annotation for the first image
    with open(annotation_output, 'r') as f:
        annotations = json.load(f)
    if annotations:
        first_ann = annotations[0]
        img_path = os.path.join(input_dir, first_ann['image_filename'])
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        vis = visualize_annotation(image_np, first_ann['labels'], output_path='data/train/bat/vis_annotated.jpg')
        print('Saved visualization to data/train/bat/vis_annotated.jpg')

if __name__ == '__main__':
    main()
