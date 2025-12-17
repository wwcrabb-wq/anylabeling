"""
Pure Python fallback implementations for Cython extensions.
Used when Cython extensions are not available or fail to import.
"""

import numpy as np
from typing import List, Tuple


def fast_nms_python(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_detections: int = 100
) -> List[int]:
    """
    Pure Python NMS implementation (fallback).
    
    Args:
        boxes: Nx4 array of [x1, y1, x2, y2] boxes
        scores: N array of confidence scores
        iou_threshold: IoU threshold for suppression
        max_detections: Maximum number of detections to return
    
    Returns:
        List of indices of kept boxes
    """
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort by scores descending
    order = np.argsort(-scores)
    suppressed = np.zeros(n, dtype=bool)
    
    keep = []
    
    for i in range(n):
        idx_i = order[i]
        
        if suppressed[idx_i]:
            continue
        
        keep.append(int(idx_i))
        
        if len(keep) >= max_detections:
            break
        
        # Get coordinates for current box
        x1_i, y1_i, x2_i, y2_i = boxes[idx_i]
        
        # Check all remaining boxes
        for j in range(i + 1, n):
            idx_j = order[j]
            
            if suppressed[idx_j]:
                continue
            
            # Get coordinates for comparison box
            x1_j, y1_j, x2_j, y2_j = boxes[idx_j]
            
            # Compute IoU
            inter_x1 = max(x1_i, x1_j)
            inter_y1 = max(y1_i, y1_j)
            inter_x2 = min(x2_i, x2_j)
            inter_y2 = min(y2_i, y2_j)
            
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            
            area_i = (x2_i - x1_i) * (y2_i - y1_i)
            area_j = (x2_j - x1_j) * (y2_j - y1_j)
            union_area = area_i + area_j - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0.0
            
            if iou > iou_threshold:
                suppressed[idx_j] = True
    
    return keep


def transform_coordinates_python(
    coords: np.ndarray,
    x_factor: float,
    y_factor: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0
) -> None:
    """
    Transform bounding box coordinates in-place (Python fallback).
    
    Args:
        coords: Nx4 array of [x1, y1, x2, y2] coordinates
        x_factor: Scaling factor for x coordinates
        y_factor: Scaling factor for y coordinates
        x_offset: Translation offset for x coordinates
        y_offset: Translation offset for y coordinates
    
    Note: Modifies coords in-place
    """
    coords[:, [0, 2]] = coords[:, [0, 2]] * x_factor + x_offset
    coords[:, [1, 3]] = coords[:, [1, 3]] * y_factor + y_offset


def normalize_image_python(
    image: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Image normalization (Python fallback).
    
    Args:
        image: Input image (H, W, C) as uint8
        mean: Channel means
        std: Channel standard deviations
    
    Returns:
        Normalized image as float32
    """
    # Convert to float and scale to [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    # Normalize with mean and std
    normalized = (normalized - mean) / std
    
    return normalized


def polygon_area_python(vertices: np.ndarray) -> float:
    """
    Calculate polygon area using Shoelace formula (Python fallback).
    
    Args:
        vertices: Nx2 array of polygon vertices
    
    Returns:
        Polygon area
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    
    return abs(area) / 2.0


def point_in_polygon_python(x: float, y: float, polygon: np.ndarray) -> bool:
    """
    Point-in-polygon test using ray casting (Python fallback).
    
    Args:
        x, y: Point coordinates
        polygon: Nx2 array of polygon vertices
    
    Returns:
        True if point is inside polygon
    """
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        
        j = i
    
    return inside


def simplify_polygon_python(polygon: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Douglas-Peucker polygon simplification (Python fallback).
    
    Args:
        polygon: Nx2 array of polygon vertices
        epsilon: Maximum allowed distance
    
    Returns:
        Simplified polygon
    """
    def point_line_distance(px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    n = len(polygon)
    if n < 3:
        return polygon
    
    # Find point with maximum distance
    max_dist = 0.0
    max_index = 0
    
    x1, y1 = polygon[0]
    x2, y2 = polygon[-1]
    
    for i in range(1, n - 1):
        px, py = polygon[i]
        dist = point_line_distance(px, py, x1, y1, x2, y2)
        if dist > max_dist:
            max_dist = dist
            max_index = i
    
    # Recursively simplify
    if max_dist > epsilon:
        left = simplify_polygon_python(polygon[:max_index+1], epsilon)
        right = simplify_polygon_python(polygon[max_index:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.array([polygon[0], polygon[-1]])


def polygon_iou_python(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """
    Calculate IoU between two polygons (Python fallback).
    
    Args:
        poly1: Nx2 array of first polygon vertices
        poly2: Mx2 array of second polygon vertices
    
    Returns:
        IoU value
    """
    area1 = polygon_area_python(poly1)
    area2 = polygon_area_python(poly2)
    
    if area1 <= 0 or area2 <= 0:
        return 0.0
    
    # Simplified bounding box IoU (for actual polygon IoU, use Shapely)
    min_x1, min_y1 = poly1.min(axis=0)
    max_x1, max_y1 = poly1.max(axis=0)
    
    min_x2, min_y2 = poly2.min(axis=0)
    max_x2, max_y2 = poly2.max(axis=0)
    
    inter_x1 = max(min_x1, min_x2)
    inter_y1 = max(min_y1, min_y2)
    inter_x2 = min(max_x1, max_x2)
    inter_y2 = min(max_y1, max_y2)
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area
