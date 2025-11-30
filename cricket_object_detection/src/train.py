"""
Model training pipeline for cricket object detection.

Uses traditional machine learning classifiers with hand-crafted features.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

from .utils import (
    TOTAL_CELLS, CLASS_LABELS, save_model, load_model,
    get_image_files
)
from .preprocess import image_to_numpy, preprocess_image
from .features import extract_grid_features, extract_cell_features
from .annotate import load_annotations_batch


def load_training_data(image_dir: str, annotations_file: str) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Load training images and their annotations.
    
    Args:
        image_dir: Directory containing training images
        annotations_file: Path to the annotations JSON file
    
    Returns:
        Tuple of (list of image arrays, list of annotation lists)
    """
    annotations = load_annotations_batch(annotations_file)
    
    images = []
    labels_list = []
    
    for ann in annotations:
        image_path = os.path.join(image_dir, ann['image_filename'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load and preprocess image
        processed = preprocess_image(image_path)
        
        if processed is None:
            print(f"Warning: Could not process: {image_path}")
            continue
        
        image_array = image_to_numpy(processed)
        images.append(image_array)
        labels_list.append(ann['labels'])
    
    print(f"Loaded {len(images)} images with annotations")
    return images, labels_list


def prepare_training_data(images: List[np.ndarray], 
                           labels_list: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and label vector for training.
    
    Args:
        images: List of image arrays
        labels_list: List of label lists (64 labels per image)
    
    Returns:
        Tuple of (feature matrix X, label vector y)
    """
    print("Extracting features from training images...")
    
    all_features = []
    all_labels = []
    
    for idx, (image, labels) in enumerate(zip(images, labels_list)):
        if (idx + 1) % 10 == 0:
            print(f"Processing image {idx + 1}/{len(images)}...")
        
        # Extract features for all 64 cells
        cell_features = extract_grid_features(image)
        
        for feature, label in zip(cell_features, labels):
            all_features.append(feature)
            all_labels.append(label)
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y


def standardize_features(X_train: np.ndarray, 
                          X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Any, Any]:
    """
    Standardize features to zero mean and unit variance.
    
    Args:
        X_train: Training feature matrix
        X_test: Optional test feature matrix
    
    Returns:
        Tuple of (standardized X_train, mean, std) or 
        (standardized X_train, standardized X_test, mean, std)
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    X_train_scaled = (X_train - mean) / std
    
    if X_test is not None:
        X_test_scaled = (X_test - mean) / std
        return X_train_scaled, X_test_scaled, mean, std
    
    return X_train_scaled, mean, std


class SimpleLogisticRegression:
    """
    Simple multi-class logistic regression using one-vs-rest strategy.
    
    Implemented from scratch without sklearn for demonstration.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: float = 0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = None
        self.biases = None
        self.classes = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit a binary classifier."""
        n_samples, n_features = X.shape
        
        # Initialize weights
        w = np.zeros(n_features)
        b = 0
        
        for _ in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, w) + b
            predictions = self._sigmoid(z)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y)) + self.regularization * w
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update weights
            w -= self.learning_rate * dw
            b -= self.learning_rate * db
        
        return w, b
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleLogisticRegression':
        """
        Fit the multi-class classifier using one-vs-rest strategy.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Label vector of shape (n_samples,)
        
        Returns:
            self
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        self.weights = np.zeros((n_classes, n_features))
        self.biases = np.zeros(n_classes)
        
        print(f"Training {n_classes} binary classifiers...")
        
        for idx, cls in enumerate(self.classes):
            # Binary labels for one-vs-rest
            y_binary = (y == cls).astype(np.float64)
            
            # Fit binary classifier
            w, b = self._fit_binary(X, y_binary)
            self.weights[idx] = w
            self.biases[idx] = b
            
            print(f"  Classifier for class {cls} trained")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            Probability matrix of shape (n_samples, n_classes)
        """
        scores = np.dot(X, self.weights.T) + self.biases
        probas = self._sigmoid(scores)
        
        # Normalize to get probabilities
        probas = probas / (probas.sum(axis=1, keepdims=True) + 1e-6)
        
        return probas
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            Predicted labels of shape (n_samples,)
        """
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]


class RandomForestClassifier:
    """
    Simple Random Forest classifier implemented from scratch.
    """
    
    def __init__(self, n_trees: int = 10, max_depth: int = 10,
                 min_samples_split: int = 5, n_features: Optional[int] = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        self.classes = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """Fit the random forest classifier."""
        self.classes = np.unique(y)
        n_samples, n_total_features = X.shape
        
        if self.n_features is None:
            self.n_features = int(np.sqrt(n_total_features))
        
        print(f"Training Random Forest with {self.n_trees} trees...")
        
        for i in range(self.n_trees):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train decision tree
            tree = DecisionTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              n_features=self.n_features)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            if (i + 1) % 5 == 0:
                print(f"  Trained {i + 1}/{self.n_trees} trees")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority voting from all trees."""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority voting
        final_predictions = []
        for i in range(X.shape[0]):
            values, counts = np.unique(predictions[:, i], return_counts=True)
            final_predictions.append(values[np.argmax(counts)])
        
        return np.array(final_predictions)


class DecisionTree:
    """
    Simple Decision Tree classifier for use in Random Forest.
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 5,
                 n_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """Fit the decision tree."""
        self.n_features = self.n_features or X.shape[1]
        self.root = self._grow_tree(X, y, depth=0)
        return self
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Dict:
        """Recursively grow the tree."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return {'leaf': True, 'value': leaf_value}
        
        # Select random features
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        
        # Find best split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return {'leaf': True, 'value': leaf_value}
        
        # Split
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = ~left_idxs
        
        if left_idxs.sum() == 0 or right_idxs.sum() == 0:
            leaf_value = self._most_common_label(y)
            return {'leaf': True, 'value': leaf_value}
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feat,
            'threshold': best_thresh,
            'left': left,
            'right': right
        }
    
    def _best_split(self, X: np.ndarray, y: np.ndarray, 
                    feat_idxs: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Find the best split."""
        best_gain = -1
        best_feat = None
        best_thresh = None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            # Sample thresholds if too many
            if len(thresholds) > 10:
                thresholds = np.percentile(X_column, np.linspace(10, 90, 10))
            
            for thresh in thresholds:
                gain = self._information_gain(y, X_column, thresh)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh
        
        return best_feat, best_thresh
    
    def _information_gain(self, y: np.ndarray, X_column: np.ndarray, 
                           threshold: float) -> float:
        """Calculate information gain for a split."""
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Split
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs
        
        if left_idxs.sum() == 0 or right_idxs.sum() == 0:
            return 0
        
        n = len(y)
        n_l, n_r = left_idxs.sum(), right_idxs.sum()
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        
        # Weighted entropy of children
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        
        return parent_entropy - child_entropy
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy."""
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        return -np.sum([p * np.log(p + 1e-10) for p in ps if p > 0])
    
    def _most_common_label(self, y: np.ndarray) -> int:
        """Return the most common label."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x: np.ndarray, node: Dict) -> int:
        """Traverse the tree to predict a single sample."""
        if node['leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])


def train_model(X: np.ndarray, y: np.ndarray, 
                model_type: str = 'random_forest') -> Tuple[Any, float, float]:
    """
    Train a classifier on the prepared data.
    
    Args:
        X: Feature matrix
        y: Label vector
        model_type: Type of model ('logistic', 'random_forest')
    
    Returns:
        Tuple of (trained model, mean, std for standardization)
    """
    # Standardize features
    X_scaled, mean, std = standardize_features(X)
    
    # Choose and train model
    if model_type == 'logistic':
        model = SimpleLogisticRegression(learning_rate=0.1, n_iterations=500)
    else:  # random_forest
        model = RandomForestClassifier(n_trees=20, max_depth=15)
    
    print(f"\nTraining {model_type} classifier...")
    model.fit(X_scaled, y)
    
    # Training accuracy
    predictions = model.predict(X_scaled)
    accuracy = np.mean(predictions == y)
    print(f"Training accuracy: {accuracy:.4f}")
    
    return model, mean, std


def cross_validate(X: np.ndarray, y: np.ndarray, n_folds: int = 5,
                   model_type: str = 'random_forest') -> float:
    """
    Perform k-fold cross-validation.
    
    Args:
        X: Feature matrix
        y: Label vector
        n_folds: Number of folds
        model_type: Type of model to use
    
    Returns:
        Mean cross-validation accuracy
    """
    n_samples = len(y)
    fold_size = n_samples // n_folds
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    accuracies = []
    
    for fold in range(n_folds):
        # Split data
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
        
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Standardize
        X_train_scaled, X_test_scaled, _, _ = standardize_features(X_train, X_test)
        
        # Train
        if model_type == 'logistic':
            model = SimpleLogisticRegression(learning_rate=0.1, n_iterations=300)
        else:
            model = RandomForestClassifier(n_trees=10, max_depth=10)
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        predictions = model.predict(X_test_scaled)
        accuracy = np.mean(predictions == y_test)
        accuracies.append(accuracy)
        
        print(f"Fold {fold + 1}/{n_folds}: accuracy = {accuracy:.4f}")
    
    mean_accuracy = np.mean(accuracies)
    print(f"\nMean cross-validation accuracy: {mean_accuracy:.4f}")
    
    return mean_accuracy


def save_trained_model(model: Any, mean: np.ndarray, std: np.ndarray,
                       team_name: str, output_dir: str) -> str:
    """
    Save the trained model with preprocessing parameters.
    
    Args:
        model: Trained classifier
        mean: Feature mean for standardization
        std: Feature std for standardization
        team_name: Team name for the model file
        output_dir: Directory to save the model
    
    Returns:
        Path to the saved model
    """
    model_data = {
        'model': model,
        'mean': mean,
        'std': std,
        'team_name': team_name
    }
    
    filepath = os.path.join(output_dir, f'model_{team_name}.pkl')
    save_model(model_data, filepath)
    
    return filepath


def train_from_directory(train_image_dir: str, annotations_file: str,
                          team_name: str, output_dir: str,
                          model_type: str = 'random_forest') -> str:
    """
    Complete training pipeline from image directory.
    
    Args:
        train_image_dir: Directory containing training images
        annotations_file: Path to annotations JSON file
        team_name: Team name for the model file
        output_dir: Directory to save the model
        model_type: Type of model to train
    
    Returns:
        Path to the saved model
    """
    # Load data
    images, labels_list = load_training_data(train_image_dir, annotations_file)
    
    if len(images) == 0:
        raise ValueError("No training images found!")
    
    # Prepare features
    X, y = prepare_training_data(images, labels_list)
    
    # Train model
    model, mean, std = train_model(X, y, model_type)
    
    # Save model
    model_path = save_trained_model(model, mean, std, team_name, output_dir)
    
    return model_path
