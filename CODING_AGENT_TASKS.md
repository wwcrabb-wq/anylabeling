# Coding Agent Implementation Roadmap

This file serves as a comprehensive instruction guide for the GitHub Copilot coding agent to implement all missing features in the AnyLabeling project. Phase 1 (Python-level optimizations) has been completed. This document covers Phases 2-4 and additional enhancements.

## Current Status

✅ **Completed (Phase 1):**
- Batch processing for model inference
- Multi-threaded image filtering
- Parallel processing utilities
- Image caching system (LRU)
- Performance monitoring utilities
- Configuration system
- Basic documentation

📋 **Remaining Work:** All sections below

---

## Section 1: Cython Extensions (Phase 2)

### 1.1 Create `anylabeling/extensions/fast_nms.pyx`

```cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Optimized Non-Maximum Suppression (NMS) using Cython.
Provides 10-50x speedup over pure Python implementations.
"""

cimport numpy as cnp
import numpy as np
from libc.math cimport fmax, fmin
cimport cython

ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.int32_t ITYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t compute_iou(
    DTYPE_t x1_1, DTYPE_t y1_1, DTYPE_t x2_1, DTYPE_t y2_1,
    DTYPE_t x1_2, DTYPE_t y1_2, DTYPE_t x2_2, DTYPE_t y2_2
) nogil:
    """
    Compute IoU between two boxes.
    
    Args:
        x1_1, y1_1, x2_1, y2_1: First box coordinates
        x1_2, y1_2, x2_2, y2_2: Second box coordinates
    
    Returns:
        IoU value between 0 and 1
    """
    cdef DTYPE_t inter_x1 = fmax(x1_1, x1_2)
    cdef DTYPE_t inter_y1 = fmax(y1_1, y1_2)
    cdef DTYPE_t inter_x2 = fmin(x2_1, x2_2)
    cdef DTYPE_t inter_y2 = fmin(y2_1, y2_2)
    
    cdef DTYPE_t inter_w = fmax(0.0, inter_x2 - inter_x1)
    cdef DTYPE_t inter_h = fmax(0.0, inter_y2 - inter_y1)
    cdef DTYPE_t inter_area = inter_w * inter_h
    
    cdef DTYPE_t area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    cdef DTYPE_t area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    cdef DTYPE_t union_area = area1 + area2 - inter_area
    
    if union_area <= 0.0:
        return 0.0
    
    return inter_area / union_area


cpdef list fast_nms_cython(
    cnp.ndarray[DTYPE_t, ndim=2] boxes,
    cnp.ndarray[DTYPE_t, ndim=1] scores,
    float iou_threshold,
    int max_detections=100
):
    """
    Optimized NMS using Cython with typed memoryviews.
    
    Args:
        boxes: Nx4 array of [x1, y1, x2, y2] boxes
        scores: N array of confidence scores
        iou_threshold: IoU threshold for suppression
        max_detections: Maximum number of detections to return
    
    Returns:
        List of indices of kept boxes
    
    Algorithm:
        1. Sort boxes by score descending
        2. For each box, suppress all boxes with IoU > threshold
        3. Use vectorized IoU computation
        4. Early termination when max_detections reached
    """
    cdef int n = boxes.shape[0]
    if n == 0:
        return []
    
    # Sort by scores descending
    cdef cnp.ndarray[ITYPE_t, ndim=1] order = np.argsort(-scores).astype(np.int32)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, cast=True] suppressed = np.zeros(n, dtype=np.uint8)
    
    cdef list keep = []
    cdef int i, j, idx_i, idx_j
    cdef DTYPE_t iou
    
    # Access box data
    cdef DTYPE_t x1_i, y1_i, x2_i, y2_i
    cdef DTYPE_t x1_j, y1_j, x2_j, y2_j
    
    for i in range(n):
        idx_i = order[i]
        
        if suppressed[idx_i]:
            continue
        
        keep.append(idx_i)
        
        if len(keep) >= max_detections:
            break
        
        # Get coordinates for current box
        x1_i = boxes[idx_i, 0]
        y1_i = boxes[idx_i, 1]
        x2_i = boxes[idx_i, 2]
        y2_i = boxes[idx_i, 3]
        
        # Check all remaining boxes
        for j in range(i + 1, n):
            idx_j = order[j]
            
            if suppressed[idx_j]:
                continue
            
            # Get coordinates for comparison box
            x1_j = boxes[idx_j, 0]
            y1_j = boxes[idx_j, 1]
            x2_j = boxes[idx_j, 2]
            y2_j = boxes[idx_j, 3]
            
            # Compute IoU
            iou = compute_iou(x1_i, y1_i, x2_i, y2_i,
                             x1_j, y1_j, x2_j, y2_j)
            
            if iou > iou_threshold:
                suppressed[idx_j] = 1
    
    return keep
```

**Implementation Notes:**
- Use typed memoryviews for maximum performance
- Enable compiler directives for optimization (boundscheck=False, wraparound=False)
- Use nogil for critical inner loops when possible
- Implement early termination when max_detections is reached
- Handle edge cases (empty arrays, all suppressed)

**Testing Requirements:**
- Test with various box counts (1, 10, 100, 1000+)
- Test with different IoU thresholds
- Test with overlapping and non-overlapping boxes
- Verify results match pure Python implementation
- Benchmark against pure Python and OpenCV NMS

### 1.2 Create `anylabeling/extensions/fast_transforms.pyx`

```cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Fast image transformation operations using Cython.
Optimized coordinate transforms and preprocessing.
"""

cimport numpy as cnp
import numpy as np
cimport cython

ctypedef cnp.float32_t DTYPE_t
ctypedef cnp.uint8_t UINT8_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void transform_coordinates_inplace(
    cnp.ndarray[DTYPE_t, ndim=2] coords,
    float x_factor,
    float y_factor,
    float x_offset=0,
    float y_offset=0
):
    """
    Transform bounding box coordinates in-place for zero-copy performance.
    
    Args:
        coords: Nx4 array of [x1, y1, x2, y2] coordinates
        x_factor: Scaling factor for x coordinates
        y_factor: Scaling factor for y coordinates
        x_offset: Translation offset for x coordinates
        y_offset: Translation offset for y coordinates
    
    Note: Modifies coords in-place for maximum performance
    """
    cdef int n = coords.shape[0]
    cdef int i
    
    for i in range(n):
        # Transform x1, y1
        coords[i, 0] = coords[i, 0] * x_factor + x_offset
        coords[i, 1] = coords[i, 1] * y_factor + y_offset
        # Transform x2, y2
        coords[i, 2] = coords[i, 2] * x_factor + x_offset
        coords[i, 3] = coords[i, 3] * y_factor + y_offset


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[UINT8_t, ndim=3] letterbox_transform(
    cnp.ndarray[UINT8_t, ndim=3] image,
    tuple target_size,
    tuple color=(114, 114, 114)
):
    """
    Fast letterbox transformation for model input.
    Maintains aspect ratio while resizing to target size.
    
    Args:
        image: Input image (H, W, C) as uint8
        target_size: Target (height, width)
        color: Padding color (R, G, B)
    
    Returns:
        Letterboxed image with target size
    """
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef int c = image.shape[2]
    cdef int target_h = target_size[0]
    cdef int target_w = target_size[1]
    
    # Calculate scaling factor
    cdef float scale = min(target_h / <float>h, target_w / <float>w)
    cdef int new_h = <int>(h * scale)
    cdef int new_w = <int>(w * scale)
    
    # Create output array filled with padding color
    cdef cnp.ndarray[UINT8_t, ndim=3] output = np.full(
        (target_h, target_w, c), color, dtype=np.uint8
    )
    
    # Calculate padding offsets to center the image
    cdef int pad_top = (target_h - new_h) // 2
    cdef int pad_left = (target_w - new_w) // 2
    
    # Resize using OpenCV (called from Python side for simplicity)
    # In practice, this would use cv2.resize before calling this function
    # This function focuses on the padding logic
    
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=3] normalize_image(
    cnp.ndarray[UINT8_t, ndim=3] image,
    cnp.ndarray[DTYPE_t, ndim=1] mean,
    cnp.ndarray[DTYPE_t, ndim=1] std
):
    """
    Fast image normalization with SIMD optimization hints.
    
    Args:
        image: Input image (H, W, C) as uint8
        mean: Channel means (C,)
        std: Channel standard deviations (C,)
    
    Returns:
        Normalized image as float32
    """
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef int c = image.shape[2]
    
    # Create output array
    cdef cnp.ndarray[DTYPE_t, ndim=3] output = np.empty(
        (h, w, c), dtype=np.float32
    )
    
    cdef int i, j, k
    cdef DTYPE_t pixel_val
    
    # Normalize each pixel
    for i in range(h):
        for j in range(w):
            for k in range(c):
                pixel_val = <DTYPE_t>image[i, j, k] / 255.0
                output[i, j, k] = (pixel_val - mean[k]) / std[k]
    
    return output
```

**Implementation Notes:**
- In-place transforms avoid memory allocation overhead
- Letterbox transform should integrate with cv2.resize
- Normalization can benefit from parallel processing
- Consider using memory views for better performance

**Testing Requirements:**
- Test coordinate transforms with various scaling factors
- Test letterbox with different aspect ratios
- Verify normalization correctness with known mean/std values
- Benchmark against pure Python/NumPy implementations

### 1.3 Create `anylabeling/extensions/polygon_ops.pyx`

```cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Fast polygon operations using Cython.
Includes area calculation, point-in-polygon tests, simplification, and IoU.
"""

cimport numpy as cnp
import numpy as np
from libc.math cimport fabs, fmin, fmax
cimport cython

ctypedef cnp.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double polygon_area(cnp.ndarray[DTYPE_t, ndim=2] vertices):
    """
    Calculate polygon area using Shoelace formula.
    O(n) complexity with minimal memory allocation.
    
    Args:
        vertices: Nx2 array of polygon vertices (x, y)
    
    Returns:
        Polygon area (always positive)
    """
    cdef int n = vertices.shape[0]
    if n < 3:
        return 0.0
    
    cdef double area = 0.0
    cdef int i
    cdef int j
    
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    
    return fabs(area) / 2.0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint point_in_polygon(
    double x, double y,
    cnp.ndarray[DTYPE_t, ndim=2] polygon
):
    """
    Fast point-in-polygon test using ray casting algorithm.
    Returns True if point is inside polygon.
    
    Args:
        x, y: Point coordinates
        polygon: Nx2 array of polygon vertices
    
    Returns:
        True if point is inside polygon, False otherwise
    """
    cdef int n = polygon.shape[0]
    cdef bint inside = False
    cdef int i, j
    cdef double xi, yi, xj, yj
    
    j = n - 1
    for i in range(n):
        xi = polygon[i, 0]
        yi = polygon[i, 1]
        xj = polygon[j, 0]
        yj = polygon[j, 1]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        
        j = i
    
    return inside


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double point_line_distance(
    double px, double py,
    double x1, double y1,
    double x2, double y2
) nogil:
    """
    Calculate perpendicular distance from point to line segment.
    """
    cdef double dx = x2 - x1
    cdef double dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # Line segment is a point
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    
    cdef double t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    
    if t < 0:
        # Closest point is start of segment
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    elif t > 1:
        # Closest point is end of segment
        return ((px - x2) ** 2 + (py - y2) ** 2) ** 0.5
    else:
        # Closest point is on the segment
        cdef double closest_x = x1 + t * dx
        cdef double closest_y = y1 + t * dy
        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] simplify_polygon(
    cnp.ndarray[DTYPE_t, ndim=2] polygon,
    double epsilon
):
    """
    Douglas-Peucker polygon simplification algorithm.
    Reduces polygon vertices while maintaining shape.
    
    Args:
        polygon: Nx2 array of polygon vertices
        epsilon: Maximum allowed distance between original and simplified polygon
    
    Returns:
        Simplified polygon as Mx2 array (M <= N)
    """
    cdef int n = polygon.shape[0]
    if n < 3:
        return polygon
    
    # Find the point with maximum distance from line start-end
    cdef double max_dist = 0.0
    cdef int max_index = 0
    cdef int i
    cdef double dist
    
    cdef double x1 = polygon[0, 0]
    cdef double y1 = polygon[0, 1]
    cdef double x2 = polygon[n-1, 0]
    cdef double y2 = polygon[n-1, 1]
    
    for i in range(1, n - 1):
        dist = point_line_distance(
            polygon[i, 0], polygon[i, 1],
            x1, y1, x2, y2
        )
        if dist > max_dist:
            max_dist = dist
            max_index = i
    
    # If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon:
        # Recursively simplify
        left = simplify_polygon(polygon[:max_index+1], epsilon)
        right = simplify_polygon(polygon[max_index:], epsilon)
        
        # Concatenate results (remove duplicate point)
        return np.vstack((left[:-1], right))
    else:
        # All points are close enough, return endpoints
        return np.array([polygon[0], polygon[n-1]], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double polygon_iou(
    cnp.ndarray[DTYPE_t, ndim=2] poly1,
    cnp.ndarray[DTYPE_t, ndim=2] poly2
):
    """
    Calculate IoU between two polygons.
    
    Args:
        poly1: Nx2 array of first polygon vertices
        poly2: Mx2 array of second polygon vertices
    
    Returns:
        IoU value between 0 and 1
    
    Note: This is a simplified implementation. For production use,
    consider using libraries like Shapely for robust polygon intersection.
    """
    cdef double area1 = polygon_area(poly1)
    cdef double area2 = polygon_area(poly2)
    
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0
    
    # For proper polygon intersection, use Shapely or similar library
    # This is a placeholder that returns approximate IoU based on bounding boxes
    cdef double min_x1 = poly1[:, 0].min()
    cdef double max_x1 = poly1[:, 0].max()
    cdef double min_y1 = poly1[:, 1].min()
    cdef double max_y1 = poly1[:, 1].max()
    
    cdef double min_x2 = poly2[:, 0].min()
    cdef double max_x2 = poly2[:, 0].max()
    cdef double min_y2 = poly2[:, 1].min()
    cdef double max_y2 = poly2[:, 1].max()
    
    cdef double inter_x1 = fmax(min_x1, min_x2)
    cdef double inter_y1 = fmax(min_y1, min_y2)
    cdef double inter_x2 = fmin(max_x1, max_x2)
    cdef double inter_y2 = fmin(max_y1, max_y2)
    
    cdef double inter_w = fmax(0.0, inter_x2 - inter_x1)
    cdef double inter_h = fmax(0.0, inter_y2 - inter_y1)
    cdef double inter_area = inter_w * inter_h
    
    cdef double union_area = area1 + area2 - inter_area
    
    if union_area <= 0.0:
        return 0.0
    
    return inter_area / union_area
```

**Implementation Notes:**
- Douglas-Peucker algorithm is recursive; consider iterative version for large polygons
- For production polygon IoU, integrate with Shapely library
- Point-in-polygon uses ray casting (odd-even rule)
- All functions optimize for speed with minimal allocations

**Testing Requirements:**
- Test polygon_area with various shapes (triangles, squares, complex polygons)
- Test point_in_polygon with edge cases (on vertex, on edge, outside)
- Test simplify_polygon with different epsilon values
- Verify polygon_iou correctness
- Benchmark against Python implementations

### 1.4 Create `anylabeling/extensions/setup_extensions.py`

```python
"""
Build script for Cython extensions.

Usage:
    python anylabeling/extensions/setup_extensions.py build_ext --inplace

This will compile the Cython extensions and place them in the extensions directory.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
import os

# Extension definitions
extensions = [
    Extension(
        "anylabeling.extensions.fast_nms",
        ["anylabeling/extensions/fast_nms.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"] if sys.platform != "win32" else ["/O2"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "anylabeling.extensions.fast_transforms",
        ["anylabeling/extensions/fast_transforms.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"] if sys.platform != "win32" else ["/O2"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "anylabeling.extensions.polygon_ops",
        ["anylabeling/extensions/polygon_ops.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"] if sys.platform != "win32" else ["/O2"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

# Compiler directives for optimization
compiler_directives = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
    "nonecheck": False,
}

if __name__ == "__main__":
    setup(
        name="anylabeling_extensions",
        ext_modules=cythonize(
            extensions,
            compiler_directives=compiler_directives,
            annotate=True,  # Generate HTML annotation files for optimization analysis
        ),
        zip_safe=False,
    )
```

**Implementation Notes:**
- Platform-specific compiler flags (GCC/Clang vs MSVC)
- Annotation files help identify optimization opportunities
- Requires Cython>=3.0.0 and a C compiler
- Use `--inplace` to build extensions in source tree

**Testing Requirements:**
- Test build on Linux, Windows, macOS
- Verify extensions are importable after build
- Test with different Python versions (3.8, 3.9, 3.10, 3.11, 3.12)

### 1.5 Create `anylabeling/extensions/fallbacks.py`

```python
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
) -> np.ndarray:
    """
    Transform bounding box coordinates (Python fallback).
    
    Args:
        coords: Nx4 array of [x1, y1, x2, y2] coordinates
        x_factor: Scaling factor for x coordinates
        y_factor: Scaling factor for y coordinates
        x_offset: Translation offset for x coordinates
        y_offset: Translation offset for y coordinates
    
    Returns:
        Transformed coordinates
    """
    result = coords.copy()
    result[:, [0, 2]] = result[:, [0, 2]] * x_factor + x_offset
    result[:, [1, 3]] = result[:, [1, 3]] * y_factor + y_offset
    return result


def letterbox_transform_python(
    image: np.ndarray,
    target_size: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """
    Letterbox transformation (Python fallback).
    
    Args:
        image: Input image (H, W, C)
        target_size: Target (height, width)
        color: Padding color (R, G, B)
    
    Returns:
        Letterboxed image
    """
    import cv2
    
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_h / h, target_w / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create output with padding
    output = np.full((target_h, target_w, image.shape[2]), color, dtype=np.uint8)
    
    # Calculate padding offsets
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    
    # Place resized image in output
    output[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    
    return output


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
```

**Implementation Notes:**
- Pure Python implementations for portability
- Match Cython function signatures exactly
- Performance will be 10-50x slower than Cython but maintains compatibility
- Use NumPy operations where possible for better performance

**Testing Requirements:**
- Verify fallback functions produce identical results to Cython versions
- Test import fallback mechanism
- Ensure all edge cases are handled

### 1.6 Update `anylabeling/extensions/__init__.py`

```python
"""
Extensions module with automatic fallback to Python implementations.

This module attempts to import compiled Cython extensions and falls back
to pure Python implementations if they are not available.

To build Cython extensions:
    python anylabeling/extensions/setup_extensions.py build_ext --inplace
"""

import logging

logger = logging.getLogger(__name__)

# Availability flags
CYTHON_NMS_AVAILABLE = False
CYTHON_TRANSFORMS_AVAILABLE = False
CYTHON_POLYGON_AVAILABLE = False


# Try importing fast_nms
try:
    from .fast_nms import fast_nms_cython as fast_nms
    CYTHON_NMS_AVAILABLE = True
    logger.info("Using Cython NMS implementation")
except ImportError:
    from .fallbacks import fast_nms_python as fast_nms
    logger.info("Cython NMS not available, using Python fallback")


# Try importing fast_transforms
try:
    from .fast_transforms import (
        transform_coordinates_inplace as transform_coordinates,
        letterbox_transform,
        normalize_image,
    )
    CYTHON_TRANSFORMS_AVAILABLE = True
    logger.info("Using Cython transform implementations")
except ImportError:
    from .fallbacks import (
        transform_coordinates_python as transform_coordinates,
        letterbox_transform_python as letterbox_transform,
        normalize_image_python as normalize_image,
    )
    logger.info("Cython transforms not available, using Python fallback")


# Try importing polygon_ops
try:
    from .polygon_ops import (
        polygon_area,
        point_in_polygon,
        simplify_polygon,
        polygon_iou,
    )
    CYTHON_POLYGON_AVAILABLE = True
    logger.info("Using Cython polygon implementations")
except ImportError:
    from .fallbacks import (
        polygon_area_python as polygon_area,
        point_in_polygon_python as point_in_polygon,
        simplify_polygon_python as simplify_polygon,
        polygon_iou_python as polygon_iou,
    )
    logger.info("Cython polygon ops not available, using Python fallback")


def extensions_available():
    """Check if any Cython extensions are available."""
    return any([
        CYTHON_NMS_AVAILABLE,
        CYTHON_TRANSFORMS_AVAILABLE,
        CYTHON_POLYGON_AVAILABLE,
    ])


def get_extension_status():
    """Get detailed status of all extensions."""
    return {
        "nms": "cython" if CYTHON_NMS_AVAILABLE else "python",
        "transforms": "cython" if CYTHON_TRANSFORMS_AVAILABLE else "python",
        "polygon_ops": "cython" if CYTHON_POLYGON_AVAILABLE else "python",
    }


# Export all functions
__all__ = [
    # Functions
    "fast_nms",
    "transform_coordinates",
    "letterbox_transform",
    "normalize_image",
    "polygon_area",
    "point_in_polygon",
    "simplify_polygon",
    "polygon_iou",
    # Status functions
    "extensions_available",
    "get_extension_status",
    # Flags
    "CYTHON_NMS_AVAILABLE",
    "CYTHON_TRANSFORMS_AVAILABLE",
    "CYTHON_POLYGON_AVAILABLE",
]
```

**Implementation Notes:**
- Graceful fallback if Cython extensions not built
- Consistent API regardless of backend
- Logging informs users which implementation is being used
- Status functions for debugging and UI display

**Testing Requirements:**
- Test import with and without compiled extensions
- Verify fallback mechanism works correctly
- Test status functions
- Ensure API consistency between Cython and Python versions


---

## Section 2: Rust Extensions (Phase 3)

### Overview
Rust extensions provide high-performance I/O operations using Rust's safety and speed. These are optional and provide 5-10x speedup for directory scanning and image loading operations.

### 2.1 Create Rust Project Structure

```bash
mkdir -p anylabeling/rust_extensions/src
```

Required files:
- `anylabeling/rust_extensions/Cargo.toml` - Rust package configuration
- `anylabeling/rust_extensions/pyproject.toml` - Maturin build configuration
- `anylabeling/rust_extensions/src/lib.rs` - Main library entry point
- `anylabeling/rust_extensions/src/image_loader.rs` - Parallel image loading
- `anylabeling/rust_extensions/src/directory_scanner.rs` - Fast directory traversal
- `anylabeling/rust_extensions/src/mmap_reader.rs` - Memory-mapped file reading
- `anylabeling/rust_extensions/README.md` - Build and usage instructions

### 2.2 Create `anylabeling/rust_extensions/Cargo.toml`

```toml
[package]
name = "anylabeling_rust"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0"
description = "High-performance Rust extensions for AnyLabeling"
authors = ["AnyLabeling Contributors"]

[lib]
name = "anylabeling_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
rayon = "1.8"
image = { version = "0.24", default-features = false, features = ["jpeg", "png", "bmp", "tiff", "webp"] }
memmap2 = "0.9"
walkdir = "2.4"
crossbeam-channel = "0.5"

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

**Implementation Notes:**
- Uses PyO3 for Python bindings
- Rayon for data parallelism
- Image crate for decoding
- Memory-mapped I/O for large files
- LTO and optimizations for maximum performance

### 2.3 Create `anylabeling/rust_extensions/src/lib.rs`

```rust
use pyo3::prelude::*;

mod image_loader;
mod directory_scanner;
mod mmap_reader;

use image_loader::load_images_parallel;
use directory_scanner::scan_image_directory;
use mmap_reader::MmapImageReader;

/// High-performance Rust extensions for AnyLabeling
#[pymodule]
fn anylabeling_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_images_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(scan_image_directory, m)?)?;
    m.add_class::<MmapImageReader>()?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
```

### 2.4 Create `anylabeling/rust_extensions/src/image_loader.rs`

```rust
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use std::path::Path;
use numpy::PyArray3;

/// Load multiple images in parallel using Rayon thread pool
#[pyfunction]
#[pyo3(signature = (paths, num_threads=None))]
pub fn load_images_parallel(
    py: Python,
    paths: Vec<String>,
    num_threads: Option<usize>,
) -> PyResult<PyObject> {
    // Configure thread pool
    let pool = if let Some(n) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create thread pool: {}", e)))?
    } else {
        rayon::ThreadPoolBuilder::new()
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create thread pool: {}", e)))?
    };

    // Load images in parallel
    let results: Vec<Option<(Vec<u8>, Vec<usize>)>> = pool.install(|| {
        paths.par_iter().map(|path| {
            load_image_from_path(path)
        }).collect()
    });

    // Convert to Python list
    let py_list = PyList::empty(py);
    for result in results {
        match result {
            Some((data, shape)) => {
                // Convert to numpy array
                let array = PyArray3::<u8>::from_vec(py, data, shape);
                py_list.append(array)?;
            }
            None => {
                py_list.append(py.None())?;
            }
        }
    }

    Ok(py_list.into())
}

fn load_image_from_path(path: &str) -> Option<(Vec<u8>, Vec<usize>)> {
    use image::GenericImageView;

    let img = image::open(path).ok()?;
    let (width, height) = img.dimensions();
    let rgb_img = img.to_rgb8();
    let shape = vec![height as usize, width as usize, 3];
    
    Some((rgb_img.into_raw(), shape))
}
```

**Implementation Notes:**
- Uses Rayon for parallel image loading
- Configurable thread pool
- Returns numpy arrays via PyO3
- Handles errors gracefully (returns None for failed loads)
- Uses image crate for decoding various formats

### 2.5 Create `anylabeling/rust_extensions/src/directory_scanner.rs`

```rust
use pyo3::prelude::*;
use rayon::prelude::*;
use walkdir::WalkDir;
use std::path::Path;

/// Scan directory for image files with parallel traversal
#[pyfunction]
#[pyo3(signature = (path, extensions=None, recursive=true))]
pub fn scan_image_directory(
    path: &str,
    extensions: Option<Vec<String>>,
    recursive: bool,
) -> PyResult<Vec<String>> {
    // Default image extensions
    let default_exts = vec![
        "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"
    ].into_iter().map(String::from).collect();
    
    let valid_extensions = extensions.unwrap_or(default_exts);
    
    // Set up walker
    let walker = if recursive {
        WalkDir::new(path).follow_links(true)
    } else {
        WalkDir::new(path).max_depth(1)
    };

    // Collect entries
    let entries: Vec<_> = walker
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .collect();

    // Filter by extension in parallel
    let image_paths: Vec<String> = entries
        .par_iter()
        .filter_map(|entry| {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if valid_extensions.iter().any(|e| e.to_lowercase() == ext_str) {
                    return path.to_string_lossy().to_string().into();
                }
            }
            None
        })
        .collect();

    // Sort for consistency
    let mut sorted_paths = image_paths;
    sorted_paths.sort();

    Ok(sorted_paths)
}
```

**Implementation Notes:**
- Uses walkdir for efficient directory traversal
- Parallel filtering with Rayon for large directories
- Configurable recursion and extensions
- Returns sorted list of absolute paths
- Handles symbolic links

### 2.6 Create `anylabeling/rust_extensions/src/mmap_reader.rs`

```rust
use pyo3::prelude::*;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Memory-mapped image reader for large files
#[pyclass]
pub struct MmapImageReader {
    mmap: Mmap,
    path: String,
}

#[pymethods]
impl MmapImageReader {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to open file {}: {}", path, e)
            ))?;
        
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    format!("Failed to memory map file: {}", e)
                ))?
        };

        Ok(MmapImageReader {
            mmap,
            path: path.to_string(),
        })
    }

    pub fn read_bytes(&self, offset: usize, length: usize) -> PyResult<Vec<u8>> {
        if offset + length > self.mmap.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Read beyond file bounds"
            ));
        }
        
        Ok(self.mmap[offset..offset + length].to_vec())
    }

    pub fn get_size(&self) -> usize {
        self.mmap.len()
    }

    pub fn get_path(&self) -> &str {
        &self.path
    }
}
```

**Implementation Notes:**
- Memory-mapped file I/O for large files
- Zero-copy reads when possible
- Safe bounds checking
- Useful for very large image files or video frames

### 2.7 Create `anylabeling/rust_extensions/pyproject.toml`

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "anylabeling_rust"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
]

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
module-name = "anylabeling.rust_extensions.anylabeling_rust"
```

**Implementation Notes:**
- Uses Maturin for building Rust extensions
- Compatible with Python 3.8+
- Installs in anylabeling.rust_extensions namespace

### 2.8 Create `anylabeling/rust_extensions/fallback.py`

```python
"""
Pure Python fallback implementations for Rust extensions.
Used when Rust extensions are not available or fail to import.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_images_parallel_python(
    paths: List[str],
    num_threads: Optional[int] = None
) -> List[Optional[np.ndarray]]:
    """
    Load multiple images in parallel using ThreadPoolExecutor (Python fallback).
    
    Args:
        paths: List of image paths to load
        num_threads: Number of worker threads (None for default)
    
    Returns:
        List of numpy arrays (None for failed loads)
    """
    import cv2
    
    def load_single_image(path):
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            return None
    
    max_workers = num_threads if num_threads is not None else min(8, len(paths))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(load_single_image, paths))
    
    return results


def scan_image_directory_python(
    path: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> List[str]:
    """
    Scan directory for image files (Python fallback).
    
    Args:
        path: Directory path to scan
        extensions: List of valid extensions (None for default)
        recursive: Whether to scan recursively
    
    Returns:
        Sorted list of image file paths
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"]
    else:
        extensions = ["." + ext.lower().lstrip(".") for ext in extensions]
    
    image_paths = []
    
    path_obj = Path(path)
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in path_obj.glob(pattern):
        if file_path.is_file():
            if any(file_path.suffix.lower() == ext for ext in extensions):
                image_paths.append(str(file_path.absolute()))
    
    return sorted(image_paths)


class MmapImageReaderPython:
    """
    Memory-mapped image reader (Python fallback).
    
    Uses simple file I/O since Python's mmap is less performant.
    """
    
    def __init__(self, path: str):
        """
        Initialize reader.
        
        Args:
            path: Path to file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        self.path = path
        self._size = os.path.getsize(path)
    
    def read_bytes(self, offset: int, length: int) -> bytes:
        """
        Read bytes from file.
        
        Args:
            offset: Byte offset
            length: Number of bytes to read
        
        Returns:
            Bytes read from file
        """
        if offset + length > self._size:
            raise IndexError("Read beyond file bounds")
        
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(length)
    
    def get_size(self) -> int:
        """Get file size in bytes."""
        return self._size
    
    def get_path(self) -> str:
        """Get file path."""
        return self.path
```

**Implementation Notes:**
- ThreadPoolExecutor for parallel loading (not as fast as Rust+Rayon)
- Uses cv2 or PIL for image decoding
- Path.glob() for directory scanning
- Simple file I/O for mmap fallback

### 2.9 Create `anylabeling/rust_extensions/__init__.py`

```python
"""
Rust extensions module with automatic fallback to Python implementations.

To build Rust extensions:
    pip install maturin
    cd anylabeling/rust_extensions
    maturin develop --release
"""

import logging

logger = logging.getLogger(__name__)

# Availability flag
RUST_AVAILABLE = False

# Try importing Rust extensions
try:
    from .anylabeling_rust import (
        load_images_parallel,
        scan_image_directory,
        MmapImageReader,
    )
    RUST_AVAILABLE = True
    logger.info("Using Rust I/O implementations")
except ImportError as e:
    logger.info(f"Rust extensions not available, using Python fallback: {e}")
    from .fallback import (
        load_images_parallel_python as load_images_parallel,
        scan_image_directory_python as scan_image_directory,
        MmapImageReaderPython as MmapImageReader,
    )


def rust_available():
    """Check if Rust extensions are available."""
    return RUST_AVAILABLE


__all__ = [
    "load_images_parallel",
    "scan_image_directory",
    "MmapImageReader",
    "rust_available",
    "RUST_AVAILABLE",
]
```

### 2.10 Build Instructions

Create `anylabeling/rust_extensions/README.md`:

```markdown
# Rust Extensions for AnyLabeling

High-performance I/O operations using Rust.

## Building

### Prerequisites
- Rust toolchain (install from https://rustup.rs/)
- Python 3.8+
- Maturin: `pip install maturin`

### Build for Development
```bash
cd anylabeling/rust_extensions
maturin develop --release
```

### Build Wheel
```bash
cd anylabeling/rust_extensions
maturin build --release
```

### Install
```bash
pip install target/wheels/anylabeling_rust-*.whl
```

## Performance
- 5-10x faster directory scanning
- 3-5x faster parallel image loading
- Minimal memory overhead with mmap

## Testing
```bash
pytest tests/test_rust_extensions.py
```
```

---

## Section 3: TensorRT/CUDA Integration (Phase 4)

### Overview
TensorRT provides 2-5x speedup over ONNX Runtime on NVIDIA GPUs through optimizations like kernel fusion, precision calibration (FP16/INT8), and layer optimization. This section is for NVIDIA GPU users only.

### 3.1 Create `anylabeling/services/auto_labeling/tensorrt_backend.py`

```python
"""
TensorRT inference backend for maximum GPU performance.
Provides 2-5x speedup over ONNX Runtime on NVIDIA GPUs.

Requirements:
    - NVIDIA GPU with CUDA support
    - TensorRT >= 8.6.0
    - pycuda >= 2022.1
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import hashlib
import pickle

logger = logging.getLogger(__name__)

# Optional imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.info("TensorRT not available")


class TensorRTBackend:
    """TensorRT inference engine wrapper."""
    
    def __init__(
        self,
        onnx_path: str,
        fp16: bool = True,
        int8: bool = False,
        max_batch_size: int = 8,
        workspace_size_gb: float = 4.0,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize TensorRT engine from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model file
            fp16: Enable FP16 precision (2x speedup, minimal accuracy loss)
            int8: Enable INT8 quantization (requires calibration data)
            max_batch_size: Maximum batch size for engine
            workspace_size_gb: GPU memory workspace in GB
            cache_dir: Directory to cache built engines
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available. Install tensorrt and pycuda.")
        
        self.onnx_path = onnx_path
        self.fp16 = fp16
        self.int8 = int8
        self.max_batch_size = max_batch_size
        self.workspace_size = int(workspace_size_gb * (1 << 30))  # Convert to bytes
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".anylabeling" / "tensorrt_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize engine
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        
        self._load_or_build_engine()
    
    def _get_engine_cache_path(self) -> Path:
        """Get cache path for engine based on model config."""
        # Create hash of model path and configuration
        config_str = f"{self.onnx_path}_{self.fp16}_{self.int8}_{self.max_batch_size}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        model_name = Path(self.onnx_path).stem
        return self.cache_dir / f"{model_name}_{config_hash}.trt"
    
    def _load_or_build_engine(self):
        """Load cached engine or build new one."""
        cache_path = self._get_engine_cache_path()
        
        if cache_path.exists():
            logger.info(f"Loading cached TensorRT engine from {cache_path}")
            try:
                self._load_engine(cache_path)
                return
            except Exception as e:
                logger.warning(f"Failed to load cached engine: {e}. Rebuilding...")
        
        logger.info("Building TensorRT engine (this may take several minutes)...")
        self.build_engine(self.onnx_path)
        
        # Save to cache
        self._save_engine(cache_path)
    
    def _load_engine(self, engine_path: Path):
        """Load serialized engine from file."""
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self._setup_bindings()
    
    def _save_engine(self, engine_path: Path):
        """Save serialized engine to file."""
        with open(engine_path, "wb") as f:
            f.write(self.engine.serialize())
        logger.info(f"Saved TensorRT engine to {engine_path}")
    
    def build_engine(self, onnx_path: str) -> None:
        """Build TensorRT engine from ONNX model."""
        logger.info("Creating TensorRT builder...")
        
        # Create builder and network
        builder = trt.Builder(trt.Logger(trt.Logger.INFO))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.INFO))
        
        # Parse ONNX model
        logger.info(f"Parsing ONNX model: {onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)
        
        if self.fp16 and builder.platform_has_fast_fp16:
            logger.info("Enabling FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)
        
        if self.int8 and builder.platform_has_fast_int8:
            logger.info("Enabling INT8 precision")
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: INT8 calibration requires additional setup
            # See Int8Calibrator class below
        
        # Set optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        
        # Get input tensor
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        
        # Set min/opt/max shapes for dynamic batching
        min_shape = tuple([1] + list(input_shape[1:]))
        opt_shape = tuple([self.max_batch_size // 2] + list(input_shape[1:]))
        max_shape = tuple([self.max_batch_size] + list(input_shape[1:]))
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Build engine
        logger.info("Building TensorRT engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Deserialize engine
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        self._setup_bindings()
        
        logger.info("TensorRT engine built successfully")
    
    def _setup_bindings(self):
        """Setup input/output bindings for inference."""
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Setup bindings
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            size = trt.volume(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})
    
    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference with TensorRT.
        
        Args:
            input_tensor: Input tensor as numpy array
        
        Returns:
            Output tensor as numpy array
        """
        # Copy input to host buffer
        np.copyto(self.inputs[0]["host"], input_tensor.ravel())
        
        # Transfer input to device
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        
        # Execute inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer output to host
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        
        # Synchronize stream
        self.stream.synchronize()
        
        # Reshape and return
        return self.outputs[0]["host"].reshape(input_tensor.shape[0], -1)
    
    def infer_batch(self, input_tensors: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run batched inference.
        
        Args:
            input_tensors: List of input tensors
        
        Returns:
            List of output tensors
        """
        batch_size = len(input_tensors)
        
        if batch_size > self.max_batch_size:
            # Split into smaller batches
            results = []
            for i in range(0, batch_size, self.max_batch_size):
                batch = input_tensors[i:i + self.max_batch_size]
                results.extend(self.infer_batch(batch))
            return results
        
        # Stack tensors into batch
        batch_tensor = np.stack(input_tensors)
        
        # Run inference
        output = self.infer(batch_tensor)
        
        # Split output into individual results
        return [output[i] for i in range(batch_size)]


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibration for TensorRT quantization."""
    
    def __init__(
        self,
        calibration_images: List[str],
        batch_size: int = 8,
        cache_file: Optional[str] = None
    ):
        """
        Initialize calibrator with calibration images.
        
        Args:
            calibration_images: List of image paths for calibration
            batch_size: Calibration batch size
            cache_file: Path to calibration cache file
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.calibration_images = calibration_images
        self.batch_size = batch_size
        self.current_index = 0
        self.cache_file = cache_file
        
        # Allocate device memory for calibration
        self.device_input = None
    
    def get_batch_size(self):
        """Return batch size for calibration."""
        return self.batch_size
    
    def get_batch(self, names):
        """
        Get next batch of calibration data.
        
        Returns:
            List of device memory pointers or None if no more batches
        """
        if self.current_index >= len(self.calibration_images):
            return None
        
        # Load batch of images
        batch_images = []
        for i in range(self.batch_size):
            idx = self.current_index + i
            if idx >= len(self.calibration_images):
                break
            
            # Load and preprocess image
            import cv2
            img = cv2.imread(self.calibration_images[idx])
            # Preprocess according to model requirements
            # This is model-specific
            batch_images.append(img)
        
        self.current_index += len(batch_images)
        
        if not batch_images:
            return None
        
        # Convert to tensor and copy to device
        batch_tensor = np.array(batch_images).astype(np.float32)
        
        if self.device_input is None:
            self.device_input = cuda.mem_alloc(batch_tensor.nbytes)
        
        cuda.memcpy_htod(self.device_input, batch_tensor)
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        """Read calibration cache if available."""
        if self.cache_file and Path(self.cache_file).exists():
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache to file."""
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                f.write(cache)
```

**Implementation Notes:**
- Caches built engines to avoid rebuild overhead
- Supports FP16 and INT8 precision modes
- Dynamic shape support for variable batch sizes
- INT8 calibration requires representative dataset
- Automatic fallback to ONNX Runtime if TensorRT unavailable

**Testing Requirements:**
- Test engine build from ONNX model
- Test FP16 mode performance and accuracy
- Test batch inference
- Verify caching works correctly
- Compare outputs with ONNX Runtime for correctness


### 3.2 Create `anylabeling/services/auto_labeling/cuda_preprocess.py`

Complete CUDA preprocessing implementation with CuPy for:
- Fused resize + normalize + transpose operations
- Custom CUDA kernels for maximum performance  
- Batched preprocessing
- Pre-allocated GPU memory buffers

See problem statement for complete implementation details.

### 3.3 Update YOLO Model Classes

Modify `yolov5.py`, `yolov8.py`, `yolov11.py` to:
- Add backend selection in `__init__()`
- Implement `_select_backend()` method
- Support TensorRT, CUDA preprocessing, ONNX-GPU, ONNX-CPU
- Auto-select best available backend
- Fall back gracefully if backend unavailable

---

## Section 4: Complete Pre-loading Integration

### 4.1 Implement Pre-loading in Model Classes

Add to `anylabeling/services/auto_labeling/model.py`:
```python
def on_next_files_changed(self, next_files):
    """
    Pre-load next images in background thread.
    
    Implementation:
    1. Start background thread if not running
    2. Add files to pre-load queue
    3. Load images into cache
    4. Limit pre-load count based on config
    """
    pass
```

### 4.2 Integrate with File Navigation

Modify `anylabeling/views/labeling/widgets/file_list_widget.py`:
- Call `model.on_next_files_changed()` on navigation
- Pass next N files based on config `preload_count`
- Handle forward/backward navigation

### 4.3 Configuration

Update `anylabeling/configs/performance.yaml`:
```yaml
performance:
  preload_enabled: true
  preload_count: 3
  preload_thread_priority: low
```

---

## Section 5: Complete Result Caching Integration

### 5.1 Add Result Caching to Image Filter Dialog

Modify `anylabeling/views/labeling/widgets/image_filter_dialog.py`:
```python
def _get_cache_key(self) -> str:
    """Generate cache key from folder + model + threshold."""
    return f"{self.folder_path}:{self.model_name}:{self.min_confidence}:{self.max_confidence}"

def _load_cached_results(self) -> Optional[List[str]]:
    """Load cached filter results if valid."""
    pass

def _save_results_to_cache(self, results: List[str]) -> None:
    """Save filter results to cache."""
    pass
```

### 5.2 Add Disk Persistence

Implement cache persistence to `~/.anylabeling/filter_cache/`:
- JSON format for metadata
- Hash-based cache keys
- Timestamp-based invalidation
- Size limits and LRU eviction

---

## Section 6: Image Filter Enhancements

### 6.1 Filter by Specific Object Classes

Add to `ImageFilterDialog`:
- Multi-select list widget for detected classes
- Radio buttons: "Any class" vs "Specific classes"
- Save/load class filter preferences
- Update filter logic to check class membership

### 6.2 Preview Thumbnails

Add thumbnail preview panel:
- QScrollArea with grid layout
- Thumbnail generation (max 200x200)
- Lazy loading on scroll
- Click to open full image
- Right-click context menu (copy path, open in explorer)

### 6.3 Detection Count Filter

Add detection count controls:
- Slider: "At least N objects" (range 0-100)
- Checkbox: "Exactly N objects"  
- Slider: "At most N objects"
- Apply to filter logic

### 6.4 Custom Filter Rules

Advanced filter builder:
- Rule builder UI (add/remove rules)
- Logic operators (AND/OR)
- Rule types: class, confidence, count, area
- Save/load filter presets to JSON
- Preset dropdown in dialog

### 6.5 Export Filter Results

Add export functionality:
- Button: "Export Results"
- Formats: TXT (paths), JSON (with detections), CSV
- Copy to clipboard option
- Show export statistics

---

## Section 7: Benchmarking Suite

### 7.1 Create `benchmarks/benchmark_inference.py`

```python
"""
Benchmark model inference performance.
Measures single image, batch, and throughput metrics.
"""

def benchmark_single_image(model, images, iterations=100):
    """Benchmark single image inference with warmup and statistics."""
    pass

def benchmark_batch(model, images, batch_sizes=[1, 2, 4, 8, 16]):
    """Benchmark batch inference at various sizes."""
    pass

def benchmark_backends(onnx_path, images):
    """Compare performance across all available backends."""
    pass

def generate_report(results, output_path):
    """Generate HTML/Markdown performance report with charts."""
    pass
```

### 7.2 Create `benchmarks/benchmark_io.py`

Benchmark image I/O:
- Sequential vs parallel loading
- PIL vs OpenCV vs image crate
- Cached vs uncached
- Various image sizes and formats

### 7.3 Create `benchmarks/benchmark_nms.py`

Benchmark NMS implementations:
- Python vs Cython vs OpenCV vs torchvision
- Various box counts and IoU thresholds
- Report speedup factors

### 7.4 Create `benchmarks/benchmark_filtering.py`

Benchmark image filtering:
- Sequential vs parallel (various thread counts)
- With/without caching
- Various dataset sizes

### 7.5 Create `benchmarks/run_benchmarks.py`

Master script:
- Run all benchmarks
- Generate comprehensive HTML report
- Include system info (CPU, GPU, RAM)
- Export to JSON for tracking over time

---

## Section 8: Performance Settings UI

### 8.1 Create Performance Settings Dialog

Create `anylabeling/views/labeling/widgets/performance_settings_dialog.py`:

```python
class PerformanceSettingsDialog(QtWidgets.QDialog):
    """Dialog for configuring performance settings."""
    
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
    
    def _setup_ui(self):
        # Backend selection (Auto, TensorRT, ONNX-GPU, ONNX-CPU)
        # Batch size slider (1-16)
        # Thread count spinner
        # Cache size slider (0-2048 MB)
        # Pre-loading enable/count
        # Extension status display (Cython, Rust, TensorRT)
        # Save/Cancel buttons
        pass
```

### 8.2 Add Menu Entry

Modify `anylabeling/app.py`:
- Add "Performance Settings..." to Tools menu
- Connect to PerformanceSettingsDialog
- Update config on save
- Restart required notification for some settings

### 8.3 Status Indicator

Add status bar widget:
- Show active backend
- Show cache hit rate
- Show extension availability
- Clickable to open performance settings

---

## Section 9: Tests

### 9.1 Unit Tests for Cython Extensions

Create `tests/test_extensions.py`:
- Test fast_nms correctness and edge cases
- Test transform functions
- Test polygon operations
- Compare Cython vs Python results
- Performance benchmarks

### 9.2 Integration Tests for Batch Processing

Create `tests/test_batch_inference.py`:
- Test batch inference on all models
- Test various batch sizes
- Test empty/None handling
- Test mixed input types

### 9.3 Performance Regression Tests

Create `tests/test_performance.py`:
- Baseline performance measurements
- Detect performance regressions (>10% slowdown)
- Run on CI with consistent hardware
- Generate performance reports

### 9.4 Test Rust Extensions

Create `tests/test_rust_extensions.py`:
- Test parallel image loading
- Test directory scanning
- Test mmap reader
- Compare Rust vs Python results

### 9.5 Test TensorRT Backend

Create `tests/test_tensorrt.py`:
- Test engine building (requires GPU)
- Test FP16 mode
- Test batch inference
- Compare accuracy with ONNX Runtime

---

## Section 10: Documentation Updates

### 10.1 Update `docs/performance_guide.md`

Add sections for:
- Cython extensions (building, usage, performance)
- Rust extensions (building, usage, performance)
- TensorRT backend (setup, FP16/INT8, benchmarks)
- CUDA preprocessing (usage, performance)
- Pre-loading system (configuration, behavior)
- Result caching (configuration, persistence)
- Image filter enhancements (all new features)
- Performance settings UI (location, options)
- Benchmarking suite (how to run, interpret results)
- Troubleshooting (build issues, runtime issues)

### 10.2 Create `docs/building_extensions.md`

Comprehensive build guide:
- Prerequisites (compilers, toolchains)
- Building Cython extensions (Linux, Windows, macOS)
- Building Rust extensions (Linux, Windows, macOS)
- Installing TensorRT (NVIDIA setup)
- Troubleshooting build errors
- Verifying installation
- Platform-specific notes

### 10.3 Update `README.md`

Add Performance section:
```markdown
## Performance Optimizations

AnyLabeling includes optional performance extensions:

- **Cython Extensions**: 10-50x faster NMS and transforms
- **Rust Extensions**: 5-10x faster I/O operations
- **TensorRT**: 2-5x faster GPU inference (NVIDIA only)
- **Batch Processing**: 2-3x faster for multiple images
- **Multi-threading**: 3-4x faster image filtering

See [Performance Guide](docs/performance_guide.md) for details.
```

### 10.4 Create `docs/tensorrt_setup.md`

TensorRT setup guide:
- NVIDIA driver installation
- CUDA toolkit installation
- TensorRT installation (pip vs tar)
- Verifying installation
- Building first engine
- Troubleshooting

### 10.5 Update Docstrings

Ensure all new code has comprehensive docstrings:
- All public functions and classes
- All parameters and return types
- Usage examples
- Performance characteristics

---

## Implementation Order

Implement features in this order for maximum impact:

1. **Cython Extensions** (Section 1) - Highest performance impact, broadly applicable
2. **Image Filter Enhancements** (Section 6) - User-requested features, immediate value
3. **Complete Pre-loading** (Section 4) - Infrastructure exists, quick win
4. **Complete Result Caching** (Section 5) - Infrastructure exists, quick win
5. **Performance Settings UI** (Section 8) - Makes settings accessible
6. **Benchmarking Suite** (Section 7) - Measure all improvements
7. **TensorRT/CUDA Integration** (Section 3) - GPU users, requires specific hardware
8. **Rust Extensions** (Section 2) - Additional I/O optimization, requires Rust toolchain
9. **Tests** (Section 9) - Ensure quality throughout
10. **Documentation** (Section 10) - Complete documentation for all features

---

## Dependencies to Add

### Core `requirements.txt`
```
# Existing dependencies remain
# Add optional performance dependencies
cython>=3.0.0; extra == 'performance'
```

### GPU `requirements-gpu.txt`
```
# Add TensorRT (NVIDIA only)
nvidia-tensorrt>=8.6.0
pycuda>=2022.1

# CuPy for CUDA preprocessing  
cupy-cuda12x>=12.0.0
```

### Development `requirements-dev.txt`
```
# Add for building extensions
cython>=3.0.0
maturin>=1.0.0

# Benchmarking
matplotlib>=3.5.0
pandas>=1.5.0
```

### Optional Rust

Users wanting Rust extensions should:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build Rust extensions
cd anylabeling/rust_extensions
maturin develop --release
```

---

## Notes for Coding Agent

### Critical Guidelines

1. **Backward Compatibility**: All new features MUST be optional and backward compatible
2. **Graceful Fallbacks**: Every compiled extension MUST have a Python fallback
3. **Error Handling**: Never crash on missing optional dependencies
4. **Logging**: Use project's logging patterns consistently
5. **Configuration**: All features configurable via YAML/UI
6. **Documentation**: Update docs for every user-facing feature

### Code Style

- Follow existing code style (check `.editorconfig` if present)
- Use type hints for all new Python code
- Add docstrings to all public functions/classes
- Keep functions focused and < 50 lines when possible
- Use descriptive variable names

### Testing Strategy

- Write tests before or alongside implementation
- Test both compiled and fallback implementations
- Include edge cases (empty inputs, large inputs, invalid inputs)
- Add performance benchmarks for optimization claims
- Run tests on multiple platforms if possible

### Performance Claims

- Always benchmark before claiming speedup
- Use realistic datasets for benchmarking
- Report speedup ranges (min/typical/max)
- Note hardware dependencies (CPU cores, GPU model)
- Compare against baseline (current implementation)

### Extension Build

- Test builds on Linux, Windows, macOS
- Provide clear error messages for missing compilers
- Document all build prerequisites
- Make extensions truly optional (app works without them)
- CI should test both with and without extensions

### GPU Features

- Check for GPU availability before use
- Provide CPU fallback automatically
- Handle out-of-memory errors gracefully
- Support multiple GPU vendors where possible
- Document GPU requirements clearly

---

## Success Criteria

Mark each as complete when:

- [ ] **Cython Extensions**
  - [ ] All 3 .pyx files implemented and tested
  - [ ] Python fallbacks match Cython functionality
  - [ ] Build script works on Linux, Windows, macOS
  - [ ] 10x+ speedup demonstrated for NMS
  - [ ] Extensions import correctly with fallback

- [ ] **Rust Extensions**
  - [ ] All Rust modules implemented
  - [ ] Python fallbacks functional
  - [ ] Maturin build works on all platforms
  - [ ] 5x+ speedup for I/O operations
  - [ ] Graceful fallback if not built

- [ ] **TensorRT Backend**
  - [ ] Engine building from ONNX works
  - [ ] FP16 mode functional with 2x+ speedup
  - [ ] INT8 calibration implemented
  - [ ] Engine caching works correctly
  - [ ] Falls back to ONNX Runtime on CPU systems

- [ ] **CUDA Preprocessing**
  - [ ] Fused operations working
  - [ ] Custom kernel implemented
  - [ ] Batch preprocessing functional
  - [ ] Speedup vs CPU preprocessing measured

- [ ] **Pre-loading System**
  - [ ] Background thread loading works
  - [ ] Integration with file navigation complete
  - [ ] Configuration respected
  - [ ] Memory usage acceptable

- [ ] **Result Caching**
  - [ ] Cache key generation correct
  - [ ] Disk persistence working
  - [ ] Invalidation logic sound
  - [ ] Cache hits improve performance

- [ ] **Image Filter Enhancements**
  - [ ] Class filtering functional
  - [ ] Thumbnail preview working
  - [ ] Detection count filters work
  - [ ] Custom rule builder functional
  - [ ] Export features implemented

- [ ] **Benchmarking Suite**
  - [ ] All benchmark scripts complete
  - [ ] HTML report generation works
  - [ ] Results are reproducible
  - [ ] Documented how to run

- [ ] **Performance Settings UI**
  - [ ] Dialog created and styled
  - [ ] All settings accessible
  - [ ] Config updated correctly
  - [ ] Status indicators working

- [ ] **Tests**
  - [ ] Unit tests for all extensions
  - [ ] Integration tests pass
  - [ ] Performance regression tests working
  - [ ] CI integration complete

- [ ] **Documentation**
  - [ ] All guides complete and accurate
  - [ ] Examples provided
  - [ ] Troubleshooting sections helpful
  - [ ] README updated

---

## Quick Start Guide for Coding Agent

When tasked with implementing from this roadmap:

1. **Read the entire section** you're implementing first
2. **Check existing code** for patterns to follow
3. **Start with tests** to clarify requirements
4. **Implement Python fallback first** to establish API
5. **Add compiled version second** following same API
6. **Benchmark the improvement** to verify claims
7. **Update documentation** for the feature
8. **Test on multiple platforms** if possible
9. **Submit PR with** implementation + tests + docs + benchmarks

---

## Maintenance and Future Work

### Known Limitations

- Polygon IoU is approximate (use Shapely for exact)
- INT8 calibration requires manual dataset preparation
- TensorRT engines are GPU-specific (not portable)
- Rust extensions require Rust toolchain
- Some features only available on Linux/GPU

### Future Enhancements

- OpenVINO backend for Intel hardware
- AMD ROCm/HIP support
- Metal Performance Shaders for Apple Silicon
- WebAssembly for browser deployment
- Distributed inference for model farms
- Auto-tuning of performance parameters

### Community Contributions

Contributions welcome for:
- Additional backend implementations
- Platform-specific optimizations
- Benchmark results on various hardware
- Bug fixes and improvements
- Documentation enhancements

---

## Appendix: Platform-Specific Notes

### Linux

- Best platform for all features
- GCC/Clang for Cython
- Easy Rust installation
- Full CUDA/TensorRT support

### Windows

- Use MSVC for Cython (included with Visual Studio)
- Rust installation straightforward
- CUDA/TensorRT supported
- Some build commands differ (use PowerShell)

### macOS

- Clang for Cython (comes with Xcode)
- Rust installation via homebrew or rustup
- No CUDA/TensorRT support (Metal alternative possible)
- Intel vs Apple Silicon considerations

---

## Contact and Support

For implementation questions:
1. Check this document first
2. Review existing similar code in the project
3. Check GitHub issues for related discussions
4. Consult the main documentation
5. Ask in project discussions/issues

---

*This roadmap is comprehensive but not exhaustive. Use judgment when implementing details. The goal is high-performance, maintainable, well-documented code that enhances AnyLabeling for all users.*
