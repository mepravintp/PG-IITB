import os
import json
from PIL import Image
import numpy as np
from src.annotate import create_annotation, save_annotations_batch, TOTAL_CELLS

def manual_annotate_image(image_path):
    print(f'Annotating: {image_path}')
    image = Image.open(image_path).convert('RGB')
    image.show()
    labels = []
    print('Enter label for each grid cell (0: no object, 1: ball, 2: bat, 3: stump)')
    for i in range(TOTAL_CELLS):
        while True:
            try:
                label = int(input(f'Cell {i+1:02d}: '))
                if label in [0, 1, 2, 3]:
                    labels.append(label)
                    break
                else:
                    print('Invalid label. Enter 0, 1, 2, or 3.')
            except ValueError:
                print('Invalid input. Enter an integer.')
    annotation = create_annotation(os.path.basename(image_path), {i+1: labels[i] for i in range(TOTAL_CELLS)})
    annotation['labels'] = labels
    return annotation

def annotate_folder(image_dir, output_file):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    annotations = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        ann = manual_annotate_image(img_path)
        annotations.append(ann)
    save_annotations_batch(annotations, output_file)
    print(f'All annotations saved to {output_file}')

if __name__ == '__main__':
    image_dir = 'data/train/bat'  # Change as needed
    output_file = 'data/train/bat/manual_annotations.json'  # Change as needed
    annotate_folder(image_dir, output_file)
