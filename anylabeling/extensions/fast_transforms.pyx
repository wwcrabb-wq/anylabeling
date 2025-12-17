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
