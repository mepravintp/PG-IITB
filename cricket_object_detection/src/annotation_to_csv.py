
import os
import csv
from pathlib import Path
import itertools

def annotation_to_csv(input_dir, annotation_dir, output_csv):
    input_dir = Path(input_dir)
    annotation_dir = Path(annotation_dir)
    rows = []

    for img_file in itertools.chain(input_dir.glob('*.jpg'), input_dir.glob('*.png')):
        img_name = img_file.name
        ann_file = annotation_dir / f"{img_file.stem}.txt"
        # Try with 'annotated_' prefix if not found
        if not ann_file.exists():
            print(f"Annotation file not found for image: {img_name}")
            continue
            # ann_file_alt = annotation_dir / f"annotated_{img_file.stem}.txt"
            # if ann_file_alt.exists():
            #     ann_file = ann_file_alt
            # else:
                
        with open(ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cell, label = line.split(',')
                # cell is like c11, c23, etc.
                if not cell.startswith('c') or len(cell) != 3:
                    continue
                cell_num = cell[1:]
                if len(cell_num) != 2 or not cell_num.isdigit():
                    continue
                cell_idx = int(cell_num) - 1
                cell_row = cell_idx // 8 + 1
                cell_col = cell_idx % 8 + 1
                rows.append([img_name, cell_row, cell_col, label])

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'cell_row', 'cell_column', 'label'])
        writer.writerows(rows)
    print(f"CSV saved to {output_csv}")

if __name__ == "__main__":
    # Example usage
    input_dir = r"C:\\Users\\pravi\\PG IITB\\cricket_object_detection\\data\\train_modified"
    annotation_dir = r"C:\\Users\\pravi\\PG IITB\\cricket_object_detection\\data\\annotations_new"
    output_csv = r"C:\\Users\\pravi\\PG IITB\\cricket_object_detection\\outputs\\annotations_cells_modified.csv"
    annotation_to_csv(input_dir, annotation_dir, output_csv)
