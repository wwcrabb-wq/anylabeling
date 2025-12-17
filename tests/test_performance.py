"""Performance regression tests."""

import pytest
import time
import tempfile
from pathlib import Path


class TestPerformanceRegressions:
    """Test for performance regressions in critical paths."""

    def test_cache_get_performance(self):
        """Test cache get operation is fast."""
        from anylabeling.utils.image_cache import ImageCache
        import numpy as np
        
        cache = ImageCache(max_memory_mb=100)
        
        # Add test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        key = "/tmp/test.jpg"
        cache.put(key, image)
        
        # Measure get time
        iterations = 1000
        start_time = time.time()
        for _ in range(iterations):
            cache.get(key)
        elapsed = time.time() - start_time
        
        # Should be very fast (< 1ms per get)
        avg_time_ms = (elapsed / iterations) * 1000
        assert avg_time_ms < 1.0, f"Cache get too slow: {avg_time_ms:.2f}ms"

    def test_cache_put_performance(self):
        """Test cache put operation is reasonable."""
        from anylabeling.utils.image_cache import ImageCache
        import numpy as np
        
        cache = ImageCache(max_memory_mb=100)
        
        # Measure put time
        iterations = 100
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(iterations)
        ]
        
        start_time = time.time()
        for i, image in enumerate(images):
            cache.put(f"/tmp/test_{i}.jpg", image)
        elapsed = time.time() - start_time
        
        # Should be reasonable (< 10ms per put)
        avg_time_ms = (elapsed / iterations) * 1000
        assert avg_time_ms < 10.0, f"Cache put too slow: {avg_time_ms:.2f}ms"

    def test_filter_cache_get_performance(self):
        """Test filter result cache get operation is fast."""
        from anylabeling.utils.image_cache import FilterResultCache
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = FilterResultCache(cache_dir=tmp_dir)
            
            # Add test data
            cache.put("/tmp/test", "model", 0.5, 1.0, None, "any", 1, [], 100)
            
            # Measure get time
            iterations = 100
            start_time = time.time()
            for _ in range(iterations):
                cache.get("/tmp/test", "model", 0.5, 1.0, None, "any", 1)
            elapsed = time.time() - start_time
            
            # Should be fast (< 10ms per get with disk I/O)
            avg_time_ms = (elapsed / iterations) * 1000
            assert avg_time_ms < 10.0, f"Filter cache get too slow: {avg_time_ms:.2f}ms"

    def test_filter_cache_key_generation_performance(self):
        """Test cache key generation is fast."""
        import hashlib
        
        # Measure key generation time
        iterations = 10000
        start_time = time.time()
        for i in range(iterations):
            key_string = f"/tmp/test|model|0.50|1.00|all|any|{i}"
            hashlib.md5(key_string.encode()).hexdigest()
        elapsed = time.time() - start_time
        
        # Should be very fast (< 0.1ms per key)
        avg_time_ms = (elapsed / iterations) * 1000
        assert avg_time_ms < 0.1, f"Key generation too slow: {avg_time_ms:.2f}ms"

    def test_config_access_performance(self):
        """Test config access is fast."""
        from anylabeling.config import get_config
        
        # Measure config access time
        iterations = 1000
        start_time = time.time()
        for _ in range(iterations):
            config = get_config()
            perf_config = config.get("performance", {})
            perf_config.get("preload_enabled", True)
        elapsed = time.time() - start_time
        
        # Should be fast (< 1ms per access)
        avg_time_ms = (elapsed / iterations) * 1000
        assert avg_time_ms < 1.0, f"Config access too slow: {avg_time_ms:.2f}ms"


class TestMemoryUsage:
    """Test memory usage is reasonable."""

    def test_cache_memory_tracking_accuracy(self):
        """Test that cache accurately tracks memory usage."""
        from anylabeling.utils.image_cache import ImageCache
        import numpy as np
        
        cache = ImageCache(max_memory_mb=100)
        
        # Add images and track memory
        total_expected = 0
        for i in range(5):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            image_size = image.nbytes
            total_expected += image_size
            
            cache.put(f"/tmp/test_{i}.jpg", image)
        
        # Memory tracking should be accurate
        assert cache.current_memory_bytes == total_expected

    def test_cache_respects_memory_limit(self):
        """Test that cache respects memory limit."""
        from anylabeling.utils.image_cache import ImageCache
        import numpy as np
        
        # Small cache
        max_mb = 1
        cache = ImageCache(max_memory_mb=max_mb)
        
        # Try to add more images than cache can hold
        for i in range(10):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cache.put(f"/tmp/test_{i}.jpg", image)
        
        # Should not exceed limit
        assert cache.current_memory_bytes <= max_mb * 1024 * 1024


class TestConcurrency:
    """Test thread safety of caches."""

    def test_image_cache_thread_safety(self):
        """Test ImageCache is thread-safe."""
        from anylabeling.utils.image_cache import ImageCache
        import numpy as np
        import threading
        
        cache = ImageCache(max_memory_mb=100)
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(10):
                    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    key = f"/tmp/thread_{thread_id}_img_{i}.jpg"
                    cache.put(key, image)
                    cache.get(key)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not have any errors
        assert len(errors) == 0

    def test_filter_cache_thread_safety(self):
        """Test FilterResultCache is thread-safe."""
        from anylabeling.utils.image_cache import FilterResultCache
        import threading
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = FilterResultCache(cache_dir=tmp_dir)
            errors = []
            
            def worker(thread_id):
                try:
                    for i in range(5):
                        folder = f"/tmp/thread_{thread_id}_folder_{i}"
                        cache.put(folder, "model", 0.5, 1.0, None, "any", 1, [], 100)
                        cache.get(folder, "model", 0.5, 1.0, None, "any", 1)
                except Exception as e:
                    errors.append(e)
            
            # Run multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # Should not have any errors
            assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
