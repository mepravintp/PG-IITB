"""
Hand-crafted feature extraction module for cricket image classification.
Implements HOG, color histograms, edge detection, and texture features.
"""

import numpy as np
from typing import List, Tuple, Optional


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    
    Args:
        image: numpy array of shape (H, W, 3)
        
    Returns:
        Grayscale image of shape (H, W)
    """
    return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def compute_gradients(gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients using Sobel-like operators.
    
    Args:
        gray_image: Grayscale image of shape (H, W)
        
    Returns:
        Tuple of (gradient_magnitude, gradient_direction)
    """
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)
    
    # Pad image
    padded = np.pad(gray_image.astype(np.float32), ((1, 1), (1, 1)), mode='edge')
    
    # Compute gradients using vectorized convolution
    h, w = gray_image.shape
    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)
    
    # Vectorized convolution using slicing
    for di in range(3):
        for dj in range(3):
            gx += padded[di:di+h, dj:dj+w] * sobel_x[di, dj]
            gy += padded[di:di+h, dj:dj+w] * sobel_y[di, dj]
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    return magnitude, direction


def extract_hog_features(cell: np.ndarray, num_bins: int = 9) -> np.ndarray:
    """
    Extract Histogram of Oriented Gradients (HOG) features from a cell.
    
    Args:
        cell: RGB image cell of shape (75, 100, 3)
        num_bins: Number of orientation bins (default: 9)
        
    Returns:
        HOG feature vector
    """
    gray = rgb_to_grayscale(cell)
    magnitude, direction = compute_gradients(gray)
    
    # Convert direction to degrees (0-180) and bin
    direction_deg = np.degrees(direction) % 180
    bin_width = 180 / num_bins
    
    # Create histogram using vectorized operations
    bin_indices = (direction_deg / bin_width).astype(int) % num_bins
    histogram = np.bincount(bin_indices.ravel(), weights=magnitude.ravel(), minlength=num_bins).astype(np.float64)
    
    # Normalize
    norm = np.linalg.norm(histogram) + 1e-6
    return histogram / norm


def extract_color_histogram(cell: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Extract color histogram features from a cell.
    
    Args:
        cell: RGB image cell of shape (75, 100, 3)
        bins: Number of bins per color channel
        
    Returns:
        Color histogram feature vector (3 * bins)
    """
    features = []
    for channel in range(3):
        hist, _ = np.histogram(cell[:, :, channel], bins=bins, range=(0, 256))
        # Normalize
        hist = hist.astype(np.float32)
        norm = np.sum(hist) + 1e-6
        hist = hist / norm
        features.extend(hist)
    
    return np.array(features)


def extract_edge_features(cell: np.ndarray) -> np.ndarray:
    """
    Extract edge-based features using gradient analysis.
    
    Args:
        cell: RGB image cell of shape (75, 100, 3)
        
    Returns:
        Edge feature vector
    """
    gray = rgb_to_grayscale(cell)
    magnitude, direction = compute_gradients(gray)
    
    # Features: mean, std, max of gradient magnitude
    # Plus directional statistics
    features = [
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude),
        np.percentile(magnitude, 75),
        np.sum(magnitude > 50) / magnitude.size,  # Edge density
        np.mean(direction),
        np.std(direction)
    ]
    
    return np.array(features)


def extract_lbp_features(cell: np.ndarray, num_points: int = 8) -> np.ndarray:
    """
    Extract Local Binary Pattern (LBP) texture features.
    
    Args:
        cell: RGB image cell of shape (75, 100, 3)
        num_points: Number of neighbors to consider
        
    Returns:
        LBP histogram feature vector
    """
    gray = rgb_to_grayscale(cell)
    
    # Simplified LBP: compare center pixel with neighbors
    # Pattern size: 3x3
    h, w = gray.shape
    lbp_image = np.zeros((h-2, w-2), dtype=np.uint8)
    
    # 8 neighbors in clockwise order
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                 (1, 1), (1, 0), (1, -1), (0, -1)]
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = gray[i, j]
            code = 0
            for bit, (di, dj) in enumerate(neighbors):
                if gray[i + di, j + dj] >= center:
                    code |= (1 << bit)
            lbp_image[i-1, j-1] = code
    
    # Create histogram
    hist, _ = np.histogram(lbp_image.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    norm = np.sum(hist) + 1e-6
    
    return hist / norm


def extract_shape_features(cell: np.ndarray) -> np.ndarray:
    """
    Extract shape-based features (aspect ratio, compactness, etc.).
    
    Args:
        cell: RGB image cell of shape (75, 100, 3)
        
    Returns:
        Shape feature vector
    """
    gray = rgb_to_grayscale(cell)
    magnitude, _ = compute_gradients(gray)
    
    # Create binary edge map
    threshold = np.percentile(magnitude, 75)
    edge_map = (magnitude > threshold).astype(np.float32)
    
    # Features based on edge distribution
    h, w = edge_map.shape
    y_coords, x_coords = np.where(edge_map > 0)
    
    if len(y_coords) == 0:
        return np.zeros(6)
    
    # Centroid
    centroid_y = np.mean(y_coords) / h
    centroid_x = np.mean(x_coords) / w
    
    # Spread
    spread_y = np.std(y_coords) / h if len(y_coords) > 1 else 0
    spread_x = np.std(x_coords) / w if len(x_coords) > 1 else 0
    
    # Edge coverage
    edge_coverage = np.sum(edge_map) / edge_map.size
    
    # Horizontal vs vertical bias
    h_bias = spread_x / (spread_y + 1e-6)
    
    return np.array([centroid_x, centroid_y, spread_x, spread_y, edge_coverage, h_bias])


def extract_color_statistics(cell: np.ndarray) -> np.ndarray:
    """
    Extract statistical color features.
    
    Args:
        cell: RGB image cell of shape (75, 100, 3)
        
    Returns:
        Color statistics feature vector
    """
    features = []
    for channel in range(3):
        channel_data = cell[:, :, channel].astype(np.float32)
        features.extend([
            np.mean(channel_data) / 255.0,
            np.std(channel_data) / 255.0,
            np.min(channel_data) / 255.0,
            np.max(channel_data) / 255.0,
            np.median(channel_data) / 255.0
        ])
    
    # Add color ratios (useful for detecting red ball, light bat, etc.)
    r, g, b = cell[:, :, 0], cell[:, :, 1], cell[:, :, 2]
    total = r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32) + 1e-6
    features.extend([
        np.mean(r / total),
        np.mean(g / total),
        np.mean(b / total)
    ])
    
    return np.array(features)


def extract_cell_features(cell: np.ndarray) -> np.ndarray:
    """
    Extract all features from a single cell.
    
    Args:
        cell: RGB image cell of shape (75, 100, 3)
        
    Returns:
        Combined feature vector
    """
    hog = extract_hog_features(cell)           # 9 features
    color_hist = extract_color_histogram(cell) # 48 features (16 bins * 3 channels)
    edge = extract_edge_features(cell)         # 7 features
    lbp = extract_lbp_features(cell)           # 256 features
    shape = extract_shape_features(cell)       # 6 features
    color_stats = extract_color_statistics(cell)  # 18 features
    
    return np.concatenate([hog, color_hist, edge, lbp, shape, color_stats])


def extract_features_from_grid(cells: List[np.ndarray]) -> np.ndarray:
    """
    Extract features from all 64 cells of an image grid.
    
    Args:
        cells: List of 64 cell arrays
        
    Returns:
        Feature matrix of shape (64, num_features)
    """
    features = []
    for cell in cells:
        cell_features = extract_cell_features(cell)
        features.append(cell_features)
    
    return np.array(features)


def get_feature_dimension() -> int:
    """
    Get the total dimension of the feature vector for one cell.
    
    Returns:
        Number of features per cell
    """
    # HOG: 9, Color Histogram: 48, Edge: 7, LBP: 256, Shape: 6, Color Stats: 18
    return 9 + 48 + 7 + 256 + 6 + 18  # = 344
