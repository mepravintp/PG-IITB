import os
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage import io, color
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
image_dir = r"C:\\Users\\pravi\\PG IITB\\cricket_object_detection\\data\\train\\Stumps"  # Change as needed
annotation_csv = r"C:\\Users\\pravi\\PG IITB\\cricket_object_detection\\outputs\\annotations_cells.csv"
cell_height, cell_width = 75, 100  # For 800x600 images, 8x8 grid

# --- STEP 1: Load Annotations ---
df = pd.read_csv(annotation_csv)

# --- STEP 2: Split Images & Extract Cell Features ---
def extract_cell_features(cell_img):
    gray = color.rgb2gray(cell_img)
    hog_feat = hog(gray, pixels_per_cell=(8,8), cells_per_block=(1,1), feature_vector=True)
    mean = gray.mean()
    std = gray.std()
    return np.concatenate([hog_feat, [mean, std]])


# Prepare to save features and labels
feature_rows = []
img_files = df['image_name'].unique()

for img_name in img_files:
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        continue
    img = io.imread(img_path)
    # Ensure image is 800x600
    if img.shape[0] != 600 or img.shape[1] != 800:
        continue
    # Get cell labels for this image
    cell_labels = df[df['image_name'] == img_name].sort_values(['cell_row', 'cell_column'])[['cell_row', 'cell_column', 'label']].values
    # Split image into cells
    idx = 0
    for r in range(8):
        for c in range(8):
            y1, y2 = r*cell_height, (r+1)*cell_height
            x1, x2 = c*cell_width, (c+1)*cell_width
            cell_img = img[y1:y2, x1:x2]
            features = extract_cell_features(cell_img)
            cell_row, cell_col, label = cell_labels[idx]
            row = {
                'image_name': img_name,
                'cell_row': cell_row,
                'cell_column': cell_col,
                'label': label
            }
            # Add feature columns
            for i, val in enumerate(features):
                row[f'feat_{i}'] = val
            feature_rows.append(row)
            idx += 1

# Save all features to CSV
features_df = pd.DataFrame(feature_rows)
features_df.to_csv(r"C:\Users\pravi\PG IITB\cricket_object_detection\outputs\cell_features.csv", index=False)
print("Saved cell features to outputs/cell_features.csv")

# Prepare X, y for training
X = features_df[[col for col in features_df.columns if col.startswith('feat_')]].values
y = features_df['label'].values

# --- STEP 3: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 4: Train Model ---

# --- STEP 4: Train Models ---
from sklearn.neighbors import KNeighborsClassifier

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

#rf_pred = rf_clf.predict(X_test)

# KNN
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

#knn_pred=knn_clf.predict(X_test)

# --- STEP 5: Evaluate ---

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Make predictions before metrics extraction
rf_pred = rf_clf.predict(X_test)
knn_pred = knn_clf.predict(X_test)

# Collect metrics for both models
def get_metrics(y_true, y_pred, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

labels = [0, 1, 2, 3]
class_names = ['no_object', 'ball', 'bat', 'stump']

rf_precision, rf_recall, rf_f1, rf_acc = get_metrics(y_test, rf_pred, labels)
knn_precision, knn_recall, knn_f1, knn_acc = get_metrics(y_test, knn_pred, labels)

# Plot metrics for comparison
import matplotlib.pyplot as plt
import numpy as np
bar_width = 0.35
index = np.arange(len(class_names))

# Precision Comparison
plt.figure(figsize=(6,4))
plt.bar(['Random Forest', 'KNN'], [rf_acc, knn_acc], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Overall Accuracy Comparison')
plt.show()

# --- STEP 6: Predict & Visualize on a Sample Image ---

# Predict & Visualize on 10 images as subplots
num_images = min(10, len(img_files))
fig, axes = plt.subplots(4, 5, figsize=(25, 10))
axes = axes.flatten()
color_map = {0:'gray', 1:'red', 2:'green', 3:'blue'}


# Visualize predictions for both models (Random Forest and KNN)
for model_name, clf in zip(["Random Forest", "KNN"], [rf_clf, knn_clf]):
    # Visualize 3 images per model
    num_vis = min(3, len(img_files))
    fig, axes = plt.subplots(1, num_vis, figsize=(18, 6))
    if num_vis == 1:
        axes = [axes]
    for idx, img_name in enumerate(img_files[:num_vis]):
        img_path = os.path.join(image_dir, img_name)
        img = io.imread(img_path)
        pred_labels = []
        for r in range(8):
            for c in range(8):
                y1, y2 = r*cell_height, (r+1)*cell_height
                x1, x2 = c*cell_width, (c+1)*cell_width
                cell_img = img[y1:y2, x1:x2]
                features = extract_cell_features(cell_img)
                pred = clf.predict([features])[0]
                pred_labels.append(pred)
        ax = axes[idx]
        ax.imshow(img)
        for r in range(8):
            for c in range(8):
                y1, y2 = r*cell_height, (r+1)*cell_height
                x1, x2 = c*cell_width, (c+1)*cell_width
                label = pred_labels[r*8 + c]
                rect = plt.Rectangle((x1, y1), cell_width, cell_height, fill=False, edgecolor=color_map[label], linewidth=2)
                ax.add_patch(rect)
        ax.set_title(f'{img_name}')
        ax.axis('off')
    plt.suptitle(f'Predicted cell labels for {num_vis} images ({model_name})')
    plt.tight_layout()
    plt.show()


