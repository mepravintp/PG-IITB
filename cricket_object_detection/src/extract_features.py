import numpy as np
import pandas as pd

def extract_features_from_csv(csv_path, output_csv):
    df = pd.read_csv(csv_path)
    features = []

    for img_name, group in df.groupby('image_name'):
        grid = np.zeros((8, 8), dtype=int)
        for _, row in group.iterrows():
            r = int(row['cell_row']) - 1
            c = int(row['cell_column']) - 1
            label = int(row['label'])
            grid[r, c] = label

        # Feature 1: count of stumps per row
        stump_per_row = (grid == 3).sum(axis=1)
        # Feature 2: count of stumps per column
        stump_per_col = (grid == 3).sum(axis=0)
        # Feature 3: total stumps
        total_stumps = (grid == 3).sum()
        # Feature 4: total balls
        total_balls = (grid == 1).sum()
        # Feature 5: total bats
        total_bats = (grid == 2).sum()
        # Feature 6: total no_object
        total_no_object = (grid == 0).sum()

        features.append({
            'image_name': img_name,
            'total_stumps': total_stumps,
            'total_balls': total_balls,
            'total_bats': total_bats,
            'total_no_object': total_no_object,
            **{f'stump_row_{i+1}': stump_per_row[i] for i in range(8)},
            **{f'stump_col_{i+1}': stump_per_col[i] for i in range(8)},
        })

    features_df = pd.DataFrame(features)
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

if __name__ == "__main__":
    csv_path = r"C:\\Users\\pravi\\PG IITB\\cricket_object_detection\\outputs\\annotations_cells.csv"
    output_csv = r"C:\\Users\\pravi\\PG IITB\\cricket_object_detection\\outputs\\features.csv"
    extract_features_from_csv(csv_path, output_csv)
