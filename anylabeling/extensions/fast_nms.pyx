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
