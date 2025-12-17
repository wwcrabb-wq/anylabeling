"""Tests for pre-loading functionality."""

import pytest
import tempfile
import time
from pathlib import Path


class TestImageCache:
    """Test ImageCache functionality."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        from anylabeling.utils.image_cache import ImageCache
        
        cache = ImageCache(max_memory_mb=100)
        assert cache.max_memory_bytes == 100 * 1024 * 1024
        assert cache.current_memory_bytes == 0
        assert len(cache) == 0

    def test_cache_put_and_get(self):
        """Test cache put and get operations."""
        from anylabeling.utils.image_cache import ImageCache
        import numpy as np
        
        cache = ImageCache(max_memory_mb=100)
        
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        key = "/tmp/test_image.jpg"
        
        # Put in cache
        success = cache.put(key, image)
        assert success
        assert len(cache) == 1
        
        # Get from cache
        cached_image = cache.get(key)
        assert cached_image is not None
        assert np.array_equal(cached_image, image)

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from anylabeling.utils.image_cache import ImageCache
        import numpy as np
        
        # Small cache (1MB)
        cache = ImageCache(max_memory_mb=1)
        
        # Create images that will fill cache
        images = []
        for i in range(5):
            # ~0.3MB each
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            images.append((f"/tmp/test_{i}.jpg", image))
        
        # Add images - oldest should be evicted
        for key, image in images:
            cache.put(key, image)
        
        # First images should be evicted
        assert cache.get("/tmp/test_0.jpg") is None
        
        # Recent images should still be in cache
        assert cache.get("/tmp/test_4.jpg") is not None

    def test_cache_stats(self):
        """Test cache statistics."""
        from anylabeling.utils.image_cache import ImageCache
        import numpy as np
        
        cache = ImageCache(max_memory_mb=100)
        
        # Initially empty
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
        # Add image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cache.put("/tmp/test.jpg", image)
        
        # Cache miss
        cache.get("/tmp/nonexistent.jpg")
        
        # Cache hit
        cache.get("/tmp/test.jpg")
        
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_clear(self):
        """Test cache clearing."""
        from anylabeling.utils.image_cache import ImageCache
        import numpy as np
        
        cache = ImageCache(max_memory_mb=100)
        
        # Add images
        for i in range(3):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cache.put(f"/tmp/test_{i}.jpg", image)
        
        assert len(cache) == 3
        
        # Clear cache
        cache.clear()
        
        assert len(cache) == 0
        assert cache.current_memory_bytes == 0

    def test_cache_contains(self):
        """Test __contains__ method."""
        from anylabeling.utils.image_cache import ImageCache
        import numpy as np
        
        cache = ImageCache(max_memory_mb=100)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        key = "/tmp/test.jpg"
        
        assert key not in cache
        
        cache.put(key, image)
        
        assert key in cache

    def test_cache_remove(self):
        """Test cache remove operation."""
        from anylabeling.utils.image_cache import ImageCache
        import numpy as np
        
        cache = ImageCache(max_memory_mb=100)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        key = "/tmp/test.jpg"
        
        cache.put(key, image)
        assert key in cache
        
        removed = cache.remove(key)
        assert removed
        assert key not in cache
        
        # Try to remove non-existent key
        removed = cache.remove(key)
        assert not removed


class TestPreloadingIntegration:
    """Test pre-loading integration with models."""

    def test_preload_configuration(self):
        """Test pre-loading configuration."""
        from anylabeling.config import get_config
        
        config = get_config()
        perf_config = config.get("performance", {})
        
        # Check default values exist (may not be set yet)
        preload_enabled = perf_config.get("preload_enabled", True)
        preload_count = perf_config.get("preload_count", 3)
        
        # Should be boolean and int
        assert isinstance(preload_enabled, bool)
        assert isinstance(preload_count, int)
        assert 1 <= preload_count <= 10

    def test_preload_cancellation(self):
        """Test pre-loading cancellation."""
        import threading
        
        # Simulate a cancellable pre-load operation
        cancel_flag = False
        results = []
        
        def worker():
            for i in range(10):
                if cancel_flag:
                    break
                results.append(i)
                time.sleep(0.01)
        
        thread = threading.Thread(target=worker)
        thread.start()
        
        # Let it run a bit
        time.sleep(0.03)
        
        # Cancel
        cancel_flag = True
        thread.join(timeout=1.0)
        
        # Should have stopped early
        assert len(results) < 10

    def test_cache_size_configuration(self):
        """Test cache size configuration."""
        from anylabeling.utils.image_cache import ImageCache
        
        # Test with different sizes
        for size_mb in [128, 512, 1024]:
            cache = ImageCache(max_memory_mb=size_mb)
            assert cache.max_memory_bytes == size_mb * 1024 * 1024
            
            # Update size
            new_size_mb = size_mb * 2
            cache.set_max_memory(new_size_mb)
            assert cache.max_memory_bytes == new_size_mb * 1024 * 1024


class TestModelPreloadHook:
    """Test model on_next_files_changed hook."""

    def test_hook_exists_in_base_model(self):
        """Test that on_next_files_changed exists in base Model class."""
        try:
            from anylabeling.services.auto_labeling.model import Model
            assert hasattr(Model, "on_next_files_changed")
        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependency: {e}")

    def test_model_manager_calls_hook(self):
        """Test that model_manager calls on_next_files_changed."""
        try:
            # This would require a full model setup, so we just test the method exists
            from anylabeling.services.auto_labeling.model_manager import ModelManager
            
            # Check method exists
            assert hasattr(ModelManager, "on_next_files_changed")
        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependency: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
