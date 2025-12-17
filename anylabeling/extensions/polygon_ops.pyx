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
from libc.math cimport fabs, fmin, fmax, sqrt
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
        return sqrt((px - x1) * (px - x1) + (py - y1) * (py - y1))
    
    cdef double t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    
    if t < 0:
        # Closest point is start of segment
        return sqrt((px - x1) * (px - x1) + (py - y1) * (py - y1))
    elif t > 1:
        # Closest point is end of segment
        return sqrt((px - x2) * (px - x2) + (py - y2) * (py - y2))
    else:
        # Closest point is on the segment
        cdef double closest_x = x1 + t * dx
        cdef double closest_y = y1 + t * dy
        return sqrt((px - closest_x) * (px - closest_x) + (py - closest_y) * (py - closest_y))


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
    
    Note: This is a simplified implementation using bounding box approximation.
    For exact polygon intersection, use libraries like Shapely.
    """
    cdef double area1 = polygon_area(poly1)
    cdef double area2 = polygon_area(poly2)
    
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0
    
    # Simplified bounding box IoU (for actual polygon IoU, use Shapely)
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
