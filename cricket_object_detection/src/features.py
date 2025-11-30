"""
Hand-crafted feature extraction for cricket object detection.

This module implements various traditional computer vision features
WITHOUT using CNNs or deep learning methods.
"""

import numpy as np
from typing import List, Tuple, Optional
from PIL import Image


def rgb_to_hsv(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to HSV color space.
    
    Args:
        rgb_image: RGB image array of shape (H, W, 3)
    
    Returns:
        HSV image array of shape (H, W, 3)
    """
    # Normalize RGB to [0, 1]
    rgb_normalized = rgb_image.astype(np.float32) / 255.0
    
    r, g, b = rgb_normalized[:,:,0], rgb_normalized[:,:,1], rgb_normalized[:,:,2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    diff = max_val - min_val
    
    # Value
    v = max_val
    
    # Saturation
    s = np.where(max_val != 0, diff / max_val, 0)
    
    # Hue
    h = np.zeros_like(max_val)
    
    mask_r = (max_val == r) & (diff != 0)
    mask_g = (max_val == g) & (diff != 0)
    mask_b = (max_val == b) & (diff != 0)
    
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
    
    # Normalize H to [0, 1]
    h = h / 360.0
    
    return np.stack([h, s, v], axis=-1)


def compute_color_histogram(image: np.ndarray, bins: int = 8) -> np.ndarray:
    """
    Compute color histogram features.
    
    Args:
        image: RGB image array of shape (H, W, 3)
        bins: Number of bins per channel
    
    Returns:
        Normalized color histogram feature vector
    """
    features = []
    
    # RGB histograms
    for channel in range(3):
        hist, _ = np.histogram(image[:,:,channel], bins=bins, range=(0, 256))
        features.extend(hist / hist.sum() if hist.sum() > 0 else hist)
    
    # HSV histograms (helpful for cricket ball detection - red color)
    hsv = rgb_to_hsv(image)
    for channel in range(3):
        hist, _ = np.histogram(hsv[:,:,channel], bins=bins, range=(0, 1))
        features.extend(hist / hist.sum() if hist.sum() > 0 else hist)
    
    return np.array(features)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    
    Args:
        image: RGB image array of shape (H, W, 3)
    
    Returns:
        Grayscale image array of shape (H, W)
    """
    return np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)


def compute_gradients(gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients using Sobel operators.
    
    Args:
        gray_image: Grayscale image array
    
    Returns:
        Tuple of (gradient magnitude, gradient direction)
    """
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)
    
    # Pad image
    padded = np.pad(gray_image, 1, mode='edge')
    
    # Convolve
    h, w = gray_image.shape
    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(patch * sobel_x)
            gy[i, j] = np.sum(patch * sobel_y)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    return magnitude, direction


def compute_hog_features(image: np.ndarray, cell_size: int = 8, 
                          num_bins: int = 9) -> np.ndarray:
    """
    Compute Histogram of Oriented Gradients (HOG) features.
    
    This is a simplified HOG implementation without block normalization.
    
    Args:
        image: RGB image array of shape (H, W, 3)
        cell_size: Size of cells for HOG computation
        num_bins: Number of orientation bins
    
    Returns:
        HOG feature vector
    """
    gray = rgb_to_grayscale(image)
    magnitude, direction = compute_gradients(gray)
    
    # Convert direction from [-pi, pi] to [0, pi]
    direction = np.abs(direction)
    direction[direction > np.pi] = 2 * np.pi - direction[direction > np.pi]
    
    h, w = gray.shape
    num_cells_y = h // cell_size
    num_cells_x = w // cell_size
    
    hog_features = []
    
    bin_size = np.pi / num_bins
    
    for cy in range(num_cells_y):
        for cx in range(num_cells_x):
            y1, y2 = cy * cell_size, (cy + 1) * cell_size
            x1, x2 = cx * cell_size, (cx + 1) * cell_size
            
            cell_mag = magnitude[y1:y2, x1:x2]
            cell_dir = direction[y1:y2, x1:x2]
            
            hist = np.zeros(num_bins)
            
            for i in range(cell_size):
                for j in range(cell_size):
                    bin_idx = int(cell_dir[i, j] / bin_size)
                    bin_idx = min(bin_idx, num_bins - 1)
                    hist[bin_idx] += cell_mag[i, j]
            
            # Normalize
            norm = np.sqrt(np.sum(hist**2) + 1e-6)
            hist = hist / norm
            
            hog_features.extend(hist)
    
    return np.array(hog_features)


def compute_lbp_features(image: np.ndarray, num_points: int = 8, 
                          radius: int = 1) -> np.ndarray:
    """
    Compute Local Binary Pattern (LBP) features.
    
    Args:
        image: RGB image array of shape (H, W, 3)
        num_points: Number of neighbors to sample
        radius: Radius for neighbor sampling
    
    Returns:
        LBP histogram feature vector
    """
    gray = rgb_to_grayscale(image)
    h, w = gray.shape
    
    # Pad image
    padded = np.pad(gray, radius, mode='edge')
    
    # Compute LBP
    lbp = np.zeros((h, w), dtype=np.uint8)
    
    # Sample points
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    dy = -radius * np.cos(angles)
    dx = radius * np.sin(angles)
    
    for i in range(h):
        for j in range(w):
            center = padded[i + radius, j + radius]
            pattern = 0
            
            for k, (d_y, d_x) in enumerate(zip(dy, dx)):
                # Bilinear interpolation
                y = i + radius + d_y
                x = j + radius + d_x
                
                y0, x0 = int(np.floor(y)), int(np.floor(x))
                y1, x1 = y0 + 1, x0 + 1
                
                wy, wx = y - y0, x - x0
                
                neighbor = (padded[y0, x0] * (1-wy) * (1-wx) +
                           padded[y0, x1] * (1-wy) * wx +
                           padded[y1, x0] * wy * (1-wx) +
                           padded[y1, x1] * wy * wx)
                
                if neighbor >= center:
                    pattern |= (1 << k)
            
            lbp[i, j] = pattern
    
    # Compute histogram
    hist, _ = np.histogram(lbp, bins=2**num_points, range=(0, 2**num_points))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-6)
    
    return hist


def compute_edge_features(image: np.ndarray) -> np.ndarray:
    """
    Compute edge-based features.
    
    These features help detect straight edges like bats and stumps.
    
    Args:
        image: RGB image array of shape (H, W, 3)
    
    Returns:
        Edge feature vector
    """
    gray = rgb_to_grayscale(image)
    magnitude, direction = compute_gradients(gray)
    
    # Edge strength statistics
    features = [
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude),
        np.percentile(magnitude, 90),
        np.percentile(magnitude, 75),
    ]
    
    # Edge direction histogram (for detecting linear structures)
    dir_hist, _ = np.histogram(direction, bins=8, range=(-np.pi, np.pi))
    dir_hist = dir_hist.astype(np.float32)
    dir_hist = dir_hist / (dir_hist.sum() + 1e-6)
    features.extend(dir_hist)
    
    # Count of strong edges (potential object boundaries)
    threshold = np.mean(magnitude) + np.std(magnitude)
    edge_ratio = np.sum(magnitude > threshold) / magnitude.size
    features.append(edge_ratio)
    
    return np.array(features)


def compute_shape_features(image: np.ndarray) -> np.ndarray:
    """
    Compute shape-based features using image moments.
    
    Helpful for detecting round objects (ball) vs elongated objects (bat, stumps).
    
    Args:
        image: RGB image array of shape (H, W, 3)
    
    Returns:
        Shape feature vector (Hu moments)
    """
    gray = rgb_to_grayscale(image)
    
    # Apply threshold to get binary-like image
    threshold = np.mean(gray)
    binary = (gray > threshold).astype(np.float32)
    
    h, w = binary.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Raw moments
    m00 = np.sum(binary) + 1e-6
    m10 = np.sum(x_coords * binary)
    m01 = np.sum(y_coords * binary)
    
    # Centroid
    cx = m10 / m00
    cy = m01 / m00
    
    # Central moments
    x_centered = x_coords - cx
    y_centered = y_coords - cy
    
    mu20 = np.sum(x_centered**2 * binary) / m00
    mu02 = np.sum(y_centered**2 * binary) / m00
    mu11 = np.sum(x_centered * y_centered * binary) / m00
    mu30 = np.sum(x_centered**3 * binary) / m00
    mu03 = np.sum(y_centered**3 * binary) / m00
    mu21 = np.sum(x_centered**2 * y_centered * binary) / m00
    mu12 = np.sum(x_centered * y_centered**2 * binary) / m00
    
    # Normalized central moments
    n = m00 ** 0.5
    n20 = mu20 / (n**2 + 1e-6)
    n02 = mu02 / (n**2 + 1e-6)
    n11 = mu11 / (n**2 + 1e-6)
    n30 = mu30 / (n**3 + 1e-6)
    n03 = mu03 / (n**3 + 1e-6)
    n21 = mu21 / (n**(5/2) + 1e-6)
    n12 = mu12 / (n**(5/2) + 1e-6)
    
    # Hu moments (first 4, which are most useful)
    hu1 = n20 + n02
    hu2 = (n20 - n02)**2 + 4*n11**2
    hu3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    hu4 = (n30 + n12)**2 + (n21 + n03)**2
    
    # Additional shape features
    # Aspect ratio
    aspect_ratio = mu20 / (mu02 + 1e-6)
    
    # Eccentricity
    lambda1 = (mu20 + mu02) / 2 + np.sqrt(4*mu11**2 + (mu20-mu02)**2) / 2
    lambda2 = (mu20 + mu02) / 2 - np.sqrt(4*mu11**2 + (mu20-mu02)**2) / 2
    eccentricity = np.sqrt(1 - lambda2 / (lambda1 + 1e-6))
    
    return np.array([hu1, hu2, hu3, hu4, aspect_ratio, eccentricity])


def compute_color_moments(image: np.ndarray) -> np.ndarray:
    """
    Compute color moment features (mean, std, skewness per channel).
    
    Args:
        image: RGB image array of shape (H, W, 3)
    
    Returns:
        Color moment feature vector
    """
    features = []
    
    for channel in range(3):
        pixels = image[:,:,channel].flatten().astype(np.float32)
        
        # Mean
        mean = np.mean(pixels)
        
        # Standard deviation
        std = np.std(pixels)
        
        # Skewness
        if std > 0:
            skewness = np.mean(((pixels - mean) / std) ** 3)
        else:
            skewness = 0
        
        features.extend([mean / 255, std / 255, skewness])
    
    return np.array(features)


def compute_texture_features(image: np.ndarray) -> np.ndarray:
    """
    Compute texture features using GLCM-like statistics.
    
    Args:
        image: RGB image array of shape (H, W, 3)
    
    Returns:
        Texture feature vector
    """
    gray = rgb_to_grayscale(image).astype(np.uint8)
    h, w = gray.shape
    
    # Quantize to reduce computation
    gray_quantized = (gray // 16).astype(np.uint8)  # 16 levels
    
    # Compute co-occurrence features (horizontal direction)
    contrast = 0
    homogeneity = 0
    energy = 0
    correlation = 0
    
    count = 0
    for i in range(h):
        for j in range(w - 1):
            p1, p2 = gray_quantized[i, j], gray_quantized[i, j+1]
            diff = abs(int(p1) - int(p2))
            
            contrast += diff ** 2
            homogeneity += 1 / (1 + diff)
            count += 1
    
    if count > 0:
        contrast /= count
        homogeneity /= count
    
    # Energy (sum of squared pixel values)
    energy = np.sum(gray.astype(np.float32)**2) / (h * w * 255**2)
    
    # Entropy
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-6)
    entropy = -np.sum(hist * np.log(hist + 1e-6))
    
    return np.array([contrast, homogeneity, energy, entropy])


def extract_cell_features(cell_image: np.ndarray) -> np.ndarray:
    """
    Extract all hand-crafted features from a single grid cell.
    
    Args:
        cell_image: RGB image array of shape (75, 100, 3)
    
    Returns:
        Combined feature vector
    """
    features = []
    
    # Color histogram (48 features: 6 channels * 8 bins)
    color_hist = compute_color_histogram(cell_image, bins=8)
    features.append(color_hist)
    
    # HOG features (simplified, depends on image size)
    # Using smaller cells for 75x100 image
    hog = compute_hog_features(cell_image, cell_size=8, num_bins=9)
    features.append(hog)
    
    # Edge features (14 features)
    edge = compute_edge_features(cell_image)
    features.append(edge)
    
    # Shape features (6 features)
    shape = compute_shape_features(cell_image)
    features.append(shape)
    
    # Color moments (9 features)
    color_moments = compute_color_moments(cell_image)
    features.append(color_moments)
    
    # Texture features (4 features)
    texture = compute_texture_features(cell_image)
    features.append(texture)
    
    # Concatenate all features
    return np.concatenate(features)


def extract_grid_features(image: np.ndarray) -> List[np.ndarray]:
    """
    Extract features from all 64 grid cells of an image.
    
    Args:
        image: RGB image array of shape (600, 800, 3)
    
    Returns:
        List of 64 feature vectors, one per grid cell
    """
    from .preprocess import split_image_to_grid
    
    cells = split_image_to_grid(image)
    features = []
    
    for cell in cells:
        cell_features = extract_cell_features(cell)
        features.append(cell_features)
    
    return features


def extract_features_for_training(images: List[np.ndarray], 
                                   annotations: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels for training.
    
    Args:
        images: List of image arrays, each of shape (600, 800, 3)
        annotations: List of annotation lists, each containing 64 labels (0-3)
    
    Returns:
        Tuple of (feature matrix, label array)
    """
    all_features = []
    all_labels = []
    
    for image, labels in zip(images, annotations):
        grid_features = extract_grid_features(image)
        
        for feature, label in zip(grid_features, labels):
            all_features.append(feature)
            all_labels.append(label)
    
    return np.array(all_features), np.array(all_labels)
