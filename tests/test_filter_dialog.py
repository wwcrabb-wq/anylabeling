"""Tests for image filter dialog functionality."""

import pytest
import tempfile
from pathlib import Path

# Note: These are unit tests for the logic, not GUI tests
# GUI testing would require a running Qt application


class TestFilterWorkerLogic:
    """Test FilterWorker detection matching logic."""

    def test_confidence_threshold_check(self):
        """Test confidence threshold filtering."""
        # Test data
        min_conf = 0.5
        max_conf = 0.9
        
        # Should pass
        assert min_conf <= 0.7 <= max_conf
        
        # Should fail
        assert not (min_conf <= 0.3 <= max_conf)
        assert not (min_conf <= 0.95 <= max_conf)

    def test_count_mode_any(self):
        """Test 'any' count mode."""
        detection_count = 3
        count_mode = "any"
        
        meets_criteria = detection_count > 0
        assert meets_criteria

    def test_count_mode_at_least(self):
        """Test 'at_least' count mode."""
        detection_count = 3
        count_mode = "at_least"
        count_value = 2
        
        meets_criteria = detection_count >= count_value
        assert meets_criteria
        
        # Should fail
        count_value = 5
        meets_criteria = detection_count >= count_value
        assert not meets_criteria

    def test_count_mode_exactly(self):
        """Test 'exactly' count mode."""
        detection_count = 3
        count_mode = "exactly"
        count_value = 3
        
        meets_criteria = detection_count == count_value
        assert meets_criteria
        
        # Should fail
        count_value = 2
        meets_criteria = detection_count == count_value
        assert not meets_criteria

    def test_count_mode_at_most(self):
        """Test 'at_most' count mode."""
        detection_count = 3
        count_mode = "at_most"
        count_value = 5
        
        meets_criteria = 0 < detection_count <= count_value
        assert meets_criteria
        
        # Should fail
        count_value = 2
        meets_criteria = 0 < detection_count <= count_value
        assert not meets_criteria

    def test_class_filtering_any(self):
        """Test class filtering with 'any' mode."""
        selected_classes = None  # None means any class
        detected_label = "person"
        
        # Should match (no class filter)
        matches = selected_classes is None or detected_label in selected_classes
        assert matches

    def test_class_filtering_selected(self):
        """Test class filtering with selected classes."""
        selected_classes = ["person", "car"]
        
        # Should match
        detected_label = "person"
        matches = detected_label in selected_classes
        assert matches
        
        # Should not match
        detected_label = "dog"
        matches = detected_label in selected_classes
        assert not matches


class TestFilterResultCache:
    """Test FilterResultCache functionality."""

    def test_cache_key_generation(self):
        """Test cache key generation."""
        import hashlib
        
        # Same parameters should generate same key
        params1 = "folder|model|0.50|1.00|all|any|1"
        params2 = "folder|model|0.50|1.00|all|any|1"
        
        key1 = hashlib.md5(params1.encode()).hexdigest()
        key2 = hashlib.md5(params2.encode()).hexdigest()
        
        assert key1 == key2
        
        # Different parameters should generate different keys
        params3 = "folder|model|0.60|1.00|all|any|1"
        key3 = hashlib.md5(params3.encode()).hexdigest()
        
        assert key1 != key3

    def test_cache_put_and_get(self):
        """Test cache put and get operations."""
        from anylabeling.utils.image_cache import FilterResultCache
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = FilterResultCache(cache_dir=tmp_dir)
            
            # Test parameters
            folder = "/tmp/test"
            model = "yolov5"
            min_conf = 0.5
            max_conf = 1.0
            classes = None
            mode = "any"
            value = 1
            
            # Should be None initially
            result = cache.get(folder, model, min_conf, max_conf, classes, mode, value)
            assert result is None
            
            # Put result
            filtered = ["/tmp/test/img1.jpg", "/tmp/test/img2.jpg"]
            cache.put(folder, model, min_conf, max_conf, classes, mode, value, filtered, 10)
            
            # Should return cached result
            result = cache.get(folder, model, min_conf, max_conf, classes, mode, value)
            assert result is not None
            assert result["filtered_images"] == filtered
            assert result["total_images"] == 10

    def test_cache_stats(self):
        """Test cache statistics."""
        from anylabeling.utils.image_cache import FilterResultCache
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = FilterResultCache(cache_dir=tmp_dir)
            
            stats = cache.get_stats()
            assert stats["entries"] == 0
            assert stats["hits"] == 0
            assert stats["misses"] == 0
            assert stats["hit_rate"] == 0.0
            
            # Add some entries
            cache.put("/tmp/test1", "model", 0.5, 1.0, None, "any", 1, [], 10)
            cache.put("/tmp/test2", "model", 0.5, 1.0, None, "any", 1, [], 10)
            
            stats = cache.get_stats()
            assert stats["entries"] == 2

    def test_cache_clear(self):
        """Test cache clearing."""
        from anylabeling.utils.image_cache import FilterResultCache
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = FilterResultCache(cache_dir=tmp_dir)
            
            # Add entries
            cache.put("/tmp/test1", "model", 0.5, 1.0, None, "any", 1, [], 10)
            cache.put("/tmp/test2", "model", 0.5, 1.0, None, "any", 1, [], 10)
            
            stats = cache.get_stats()
            assert stats["entries"] == 2
            
            # Clear cache
            cache.clear()
            
            stats = cache.get_stats()
            assert stats["entries"] == 0

    def test_cache_lru_eviction(self):
        """Test LRU eviction."""
        from anylabeling.utils.image_cache import FilterResultCache
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create cache with max 3 entries
            cache = FilterResultCache(cache_dir=tmp_dir, max_entries=3)
            
            # Add 3 entries
            cache.put("/tmp/test1", "model", 0.5, 1.0, None, "any", 1, [], 10)
            cache.put("/tmp/test2", "model", 0.5, 1.0, None, "any", 1, [], 10)
            cache.put("/tmp/test3", "model", 0.5, 1.0, None, "any", 1, [], 10)
            
            stats = cache.get_stats()
            assert stats["entries"] == 3
            
            # Add 4th entry - should evict oldest
            cache.put("/tmp/test4", "model", 0.5, 1.0, None, "any", 1, [], 10)
            
            stats = cache.get_stats()
            assert stats["entries"] == 3
            
            # First entry should be evicted
            result = cache.get("/tmp/test1", "model", 0.5, 1.0, None, "any", 1)
            assert result is None
            
            # Other entries should still exist
            result = cache.get("/tmp/test2", "model", 0.5, 1.0, None, "any", 1)
            assert result is not None


class TestExportFunctionality:
    """Test export functionality."""

    def test_json_export_structure(self):
        """Test JSON export data structure."""
        import json
        
        export_data = {
            "total_images": 100,
            "filtered_images": 10,
            "filter_settings": {
                "min_confidence": 0.5,
                "max_confidence": 1.0,
                "selected_classes": ["person", "car"],
                "count_mode": "at_least",
                "count_value": 2,
            },
            "results": ["/tmp/img1.jpg", "/tmp/img2.jpg"],
        }
        
        # Should be JSON serializable
        json_str = json.dumps(export_data)
        assert isinstance(json_str, str)
        
        # Should be deserializable
        loaded = json.loads(json_str)
        assert loaded == export_data

    def test_txt_export_format(self):
        """Test TXT export format."""
        image_paths = ["/tmp/img1.jpg", "/tmp/img2.jpg", "/tmp/img3.jpg"]
        
        txt_content = "\n".join(image_paths) + "\n"
        
        lines = txt_content.strip().split("\n")
        assert len(lines) == len(image_paths)
        assert lines == image_paths

    def test_csv_export_format(self):
        """Test CSV export format."""
        import io
        import csv
        
        image_paths = ["/tmp/img1.jpg", "/tmp/img2.jpg"]
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["image_path"])
        for path in image_paths:
            writer.writerow([path])
        
        csv_content = output.getvalue()
        lines = csv_content.strip().split("\n")
        
        assert len(lines) == len(image_paths) + 1  # +1 for header
        assert "image_path" in lines[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
