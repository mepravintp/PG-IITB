"""
Classification model module for cricket image classification.
Implements the grid-based classifier using hand-crafted features.
"""

import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
from collections import Counter


# Label constants
LABEL_NO_OBJECT = 0
LABEL_BALL = 1
LABEL_BAT = 2
LABEL_STUMP = 3

LABEL_NAMES = {
    0: "no_object",
    1: "ball",
    2: "bat",
    3: "stump"
}


class GridClassifier:
    """
    A classifier for predicting object presence in grid cells.
    Uses a multi-class classification approach with hand-crafted features.
    """
    
    def __init__(self, n_classes: int = 4, learning_rate: float = 0.01, 
                 n_iterations: int = 1000, regularization: float = 0.01):
        """
        Initialize the classifier.
        
        Args:
            n_classes: Number of classes (4: no_object, ball, bat, stump)
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of training iterations
            regularization: L2 regularization strength
        """
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.feature_mean = None
        self.feature_std = None
        self.is_fitted = False
    
    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features using z-score normalization.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            fit: Whether to fit the normalization parameters
            
        Returns:
            Normalized feature matrix
        """
        if fit:
            self.feature_mean = np.mean(X, axis=0)
            self.feature_std = np.std(X, axis=0) + 1e-8
        
        return (X - self.feature_mean) / self.feature_std
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Apply softmax function.
        
        Args:
            z: Input logits of shape (n_samples, n_classes)
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """
        One-hot encode labels.
        
        Args:
            y: Labels of shape (n_samples,)
            
        Returns:
            One-hot encoded matrix of shape (n_samples, n_classes)
        """
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y.astype(int)] = 1
        return one_hot
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> "GridClassifier":
        """
        Train the classifier using softmax regression with gradient descent.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels of shape (n_samples,)
            verbose: Whether to print training progress
            
        Returns:
            self
        """
        # Normalize features
        X_norm = self._normalize_features(X, fit=True)
        
        n_samples, n_features = X_norm.shape
        
        # Initialize weights
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros(self.n_classes)
        
        # One-hot encode labels
        y_onehot = self._one_hot_encode(y)
        
        # Training loop
        for i in range(self.n_iterations):
            # Forward pass
            logits = np.dot(X_norm, self.weights) + self.bias
            probs = self._softmax(logits)
            
            # Compute loss (cross-entropy + L2 regularization)
            epsilon = 1e-10
            loss = -np.mean(np.sum(y_onehot * np.log(probs + epsilon), axis=1))
            loss += 0.5 * self.regularization * np.sum(self.weights ** 2)
            
            # Backward pass
            error = probs - y_onehot
            grad_weights = np.dot(X_norm.T, error) / n_samples
            grad_weights += self.regularization * self.weights
            grad_bias = np.mean(error, axis=0)
            
            # Update weights
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias
            
            if verbose and (i + 1) % 100 == 0:
                accuracy = np.mean(np.argmax(probs, axis=1) == y)
                print(f"Iteration {i+1}/{self.n_iterations}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        X_norm = self._normalize_features(X)
        logits = np.dot(X_norm, self.weights) + self.bias
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Probability matrix of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        X_norm = self._normalize_features(X)
        logits = np.dot(X_norm, self.weights) + self.bias
        return self._softmax(logits)
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a pickle file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'n_classes': self.n_classes,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "GridClassifier":
        """
        Load a model from a pickle file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded GridClassifier instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls(n_classes=model_data['n_classes'])
        classifier.weights = model_data['weights']
        classifier.bias = model_data['bias']
        classifier.feature_mean = model_data['feature_mean']
        classifier.feature_std = model_data['feature_std']
        classifier.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")
        return classifier


class RandomForestSimple:
    """
    A simplified Random Forest classifier implementation.
    Uses decision stumps with feature subsampling.
    """
    
    def __init__(self, n_trees: int = 50, max_depth: int = 5, 
                 min_samples_split: int = 5, n_features_sample: Optional[int] = None):
        """
        Initialize the Random Forest.
        
        Args:
            n_trees: Number of trees
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum samples required to split
            n_features_sample: Number of features to consider for each split
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_sample = n_features_sample
        self.trees = []
        self.is_fitted = False
        self.feature_mean = None
        self.feature_std = None
    
    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features."""
        if fit:
            self.feature_mean = np.mean(X, axis=0)
            self.feature_std = np.std(X, axis=0) + 1e-8
        return (X - self.feature_mean) / self.feature_std
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict:
        """Build a decision tree recursively."""
        n_samples, n_features = X.shape
        
        # Check stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            # Return leaf node with majority class
            if len(y) == 0:
                return {'leaf': True, 'class': 0}
            counts = Counter(y)
            return {'leaf': True, 'class': counts.most_common(1)[0][0]}
        
        # Select random features
        n_feat_select = self.n_features_sample or int(np.sqrt(n_features))
        feature_indices = np.random.choice(n_features, min(n_feat_select, n_features), replace=False)
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None
        
        current_impurity = self._gini_impurity(y)
        
        for feat_idx in feature_indices:
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds[::max(1, len(thresholds) // 10)]:  # Sample thresholds
                left_mask = X[:, feat_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue
                
                # Calculate information gain
                left_impurity = self._gini_impurity(y[left_mask])
                right_impurity = self._gini_impurity(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold
                    best_left_idx = left_mask
                    best_right_idx = right_mask
        
        if best_gain <= 0 or best_feature is None:
            counts = Counter(y)
            return {'leaf': True, 'class': counts.most_common(1)[0][0]}
        
        # Build subtrees
        left_tree = self._build_tree(X[best_left_idx], y[best_left_idx], depth + 1)
        right_tree = self._build_tree(X[best_right_idx], y[best_right_idx], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _predict_tree(self, tree: Dict, x: np.ndarray) -> int:
        """Predict using a single tree."""
        if tree['leaf']:
            return tree['class']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_tree(tree['left'], x)
        else:
            return self._predict_tree(tree['right'], x)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> "RandomForestSimple":
        """Train the random forest."""
        X_norm = self._normalize_features(X, fit=True)
        n_samples = X_norm.shape[0]
        
        self.trees = []
        for i in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X_norm[indices]
            y_sample = y[indices]
            
            tree = self._build_tree(X_sample, y_sample)
            self.trees.append(tree)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Built tree {i+1}/{self.n_trees}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        X_norm = self._normalize_features(X)
        predictions = []
        
        for x in X_norm:
            tree_preds = [self._predict_tree(tree, x) for tree in self.trees]
            predictions.append(Counter(tree_preds).most_common(1)[0][0])
        
        return np.array(predictions)
    
    def save(self, filepath: str) -> None:
        """Save the model to a pickle file."""
        model_data = {
            'trees': self.trees,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "RandomForestSimple":
        """Load a model from a pickle file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls()
        classifier.trees = model_data['trees']
        classifier.feature_mean = model_data['feature_mean']
        classifier.feature_std = model_data['feature_std']
        classifier.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")
        return classifier


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    accuracy = np.mean(y_true == y_pred)
    
    # Per-class metrics
    metrics = {'overall_accuracy': accuracy}
    
    for class_id in range(4):
        # Precision, recall for each class
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics[f'{LABEL_NAMES[class_id]}_precision'] = precision
        metrics[f'{LABEL_NAMES[class_id]}_recall'] = recall
        metrics[f'{LABEL_NAMES[class_id]}_f1'] = f1
    
    return metrics
