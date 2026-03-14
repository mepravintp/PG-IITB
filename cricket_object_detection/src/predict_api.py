import os
import numpy as np
from skimage import io, color
from flask import Flask, request, jsonify
import joblib

# Load the trained model
MODEL_PATH = r"C:\Users\pravi\PG IITB\cricket_object_detection\outputs\rf_clf_model.pkl"
model = joblib.load(MODEL_PATH)

# Feature extraction function (should match notebook)
def extract_cell_features(cell_img):
    gray = color.rgb2gray(cell_img)
    from skimage.feature import hog
    hog_feat = hog(gray, pixels_per_cell=(8,8), cells_per_block=(1,1), feature_vector=True)
    mean = gray.mean()
    std = gray.std()
    return np.concatenate([hog_feat, [mean, std]])

# API setup
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    img = io.imread(file)
    if img.shape[0] != 600 or img.shape[1] != 800:
        return jsonify({'error': 'Image must be 800x600'}), 400
    cell_height, cell_width = 75, 100
    preds = []
    for r in range(8):
        for c in range(8):
            y1, y2 = r*cell_height, (r+1)*cell_height
            x1, x2 = c*cell_width, (c+1)*cell_width
            cell_img = img[y1:y2, x1:x2]
            features = extract_cell_features(cell_img)
            pred = model.predict([features])[0]
            preds.append(int(pred))
    # Map numeric labels to class names
    label_map = {0: 'no_object', 1: 'ball', 2: 'bat', 3: 'stump'}
    pred_labels = [label_map.get(p, 'unknown') for p in preds]
    # Optionally, return a grid or just counts
    result = {
        'predictions': pred_labels,
        'bat_count': pred_labels.count('bat'),
        'ball_count': pred_labels.count('ball'),
        'stump_count': pred_labels.count('stump')
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
