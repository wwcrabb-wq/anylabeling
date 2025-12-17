"""
Unit tests for Cython extensions.
Tests correctness and compares with Python fallback implementations.
"""

import pytest
import numpy as np
from anylabeling.extensions import (
    fast_nms,
    transform_coordinates,
    normalize_image,
    polygon_area,
    point_in_polygon,
    simplify_polygon,
    polygon_iou,
    CYTHON_NMS_AVAILABLE,
    CYTHON_TRANSFORMS_AVAILABLE,
    CYTHON_POLYGON_AVAILABLE,
)
from anylabeling.extensions.fallbacks import (
    fast_nms_python,
    transform_coordinates_python,
    normalize_image_python,
    polygon_area_python,
    point_in_polygon_python,
    simplify_polygon_python,
    polygon_iou_python,
)


class TestNMS:
    """Test NMS implementations."""
    
    def test_empty_input(self):
        """Test with empty arrays."""
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)
        
        result = fast_nms(boxes, scores, 0.5)
        assert result == []
    
    def test_single_box(self):
        """Test with single box."""
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        
        result = fast_nms(boxes, scores, 0.5)
        assert result == [0]
    
    def test_non_overlapping_boxes(self):
        """Test with non-overlapping boxes."""
        boxes = np.array([
            [10, 10, 30, 30],
            [50, 50, 70, 70],
            [90, 90, 110, 110],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        
        result = fast_nms(boxes, scores, 0.5)
        assert len(result) == 3
    
    def test_overlapping_boxes(self):
        """Test with overlapping boxes."""
        boxes = np.array([
            [10, 10, 50, 50],
            [15, 15, 55, 55],  # Overlaps with first
            [90, 90, 130, 130],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        
        result = fast_nms(boxes, scores, 0.5)
        assert len(result) == 2  # Second box should be suppressed
        assert 0 in result  # Highest score kept
        assert 2 in result  # Non-overlapping kept
    
    def test_consistency_with_python(self):
        """Test that Cython and Python versions give same results."""
        np.random.seed(42)
        boxes = np.random.rand(50, 4).astype(np.float32) * 100
        boxes[:, 2] = boxes[:, 0] + np.random.rand(50) * 50
        boxes[:, 3] = boxes[:, 1] + np.random.rand(50) * 50
        scores = np.random.rand(50).astype(np.float32)
        
        result_fast = fast_nms(boxes, scores, 0.5)
        result_python = fast_nms_python(boxes, scores, 0.5)
        
        assert set(result_fast) == set(result_python)


class TestTransforms:
    """Test coordinate transform implementations."""
    
    def test_transform_coordinates(self):
        """Test coordinate transformation."""
        coords = np.array([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ], dtype=np.float32)
        
        coords_copy = coords.copy()
        transform_coordinates(coords_copy, 2.0, 2.0, 5.0, 10.0)
        
        expected = coords * np.array([2.0, 2.0, 2.0, 2.0]) + np.array([5.0, 10.0, 5.0, 10.0])
        np.testing.assert_array_almost_equal(coords_copy, expected)
    
    def test_normalize_image(self):
        """Test image normalization."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        normalized = normalize_image(image, mean, std)
        
        assert normalized.shape == image.shape
        assert normalized.dtype == np.float32
        # Check value range is reasonable after normalization
        assert -5.0 < normalized.min() < 5.0
        assert -5.0 < normalized.max() < 5.0


class TestPolygonOps:
    """Test polygon operation implementations."""
    
    def test_polygon_area_triangle(self):
        """Test area calculation for triangle."""
        vertices = np.array([
            [0, 0],
            [4, 0],
            [2, 3],
        ], dtype=np.float64)
        
        area = polygon_area(vertices)
        expected = 6.0  # Area of triangle: 0.5 * base * height = 0.5 * 4 * 3
        assert abs(area - expected) < 0.01
    
    def test_polygon_area_square(self):
        """Test area calculation for square."""
        vertices = np.array([
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ], dtype=np.float64)
        
        area = polygon_area(vertices)
        expected = 100.0
        assert abs(area - expected) < 0.01
    
    def test_point_in_polygon(self):
        """Test point-in-polygon detection."""
        square = np.array([
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ], dtype=np.float64)
        
        # Inside
        assert point_in_polygon(5, 5, square) == True
        
        # Outside
        assert point_in_polygon(15, 15, square) == False
        assert point_in_polygon(-5, 5, square) == False
    
    def test_simplify_polygon(self):
        """Test polygon simplification."""
        # Create a polygon with redundant points
        polygon = np.array([
            [0, 0],
            [5, 0.1],  # Nearly on line
            [10, 0],
            [10, 10],
            [0, 10],
        ], dtype=np.float64)
        
        simplified = simplify_polygon(polygon, epsilon=1.0)
        
        # Should remove the middle point
        assert len(simplified) <= len(polygon)
    
    def test_consistency_with_python(self):
        """Test consistency between Cython and Python implementations."""
        np.random.seed(42)
        vertices = np.random.rand(10, 2).astype(np.float64) * 100
        
        area_fast = polygon_area(vertices)
        area_python = polygon_area_python(vertices)
        
        assert abs(area_fast - area_python) < 0.01


@pytest.mark.skipif(not CYTHON_NMS_AVAILABLE, reason="Cython NMS not available")
def test_cython_nms_performance():
    """Benchmark test to ensure Cython is faster than Python."""
    import time
    
    np.random.seed(42)
    boxes = np.random.rand(1000, 4).astype(np.float32) * 100
    boxes[:, 2] = boxes[:, 0] + np.random.rand(1000) * 50
    boxes[:, 3] = boxes[:, 1] + np.random.rand(1000) * 50
    scores = np.random.rand(1000).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        fast_nms(boxes, scores, 0.5)
        fast_nms_python(boxes, scores, 0.5)
    
    # Benchmark Cython
    start = time.perf_counter()
    for _ in range(10):
        fast_nms(boxes, scores, 0.5)
    cython_time = time.perf_counter() - start
    
    # Benchmark Python
    start = time.perf_counter()
    for _ in range(10):
        fast_nms_python(boxes, scores, 0.5)
    python_time = time.perf_counter() - start
    
    speedup = python_time / cython_time
    print(f"\nCython NMS speedup: {speedup:.2f}x")
    
    # Cython should be at least 2x faster
    assert speedup > 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
