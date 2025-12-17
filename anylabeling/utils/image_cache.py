"""Image caching utilities with LRU eviction policy."""

import logging
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ImageCache:
    """LRU cache for loaded images with configurable memory limits."""

    def __init__(self, max_memory_mb: int = 512):
        """
        Initialize image cache.

        Args:
            max_memory_mb: Maximum memory usage in megabytes
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        logger.info("ImageCache initialized with %dMB limit", max_memory_mb)

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get image from cache.

        Args:
            key: Cache key (usually file path)

        Returns:
            Cached image array or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]["data"]
            self._misses += 1
            return None

    def put(self, key: str, image: np.ndarray) -> bool:
        """
        Put image in cache.

        Args:
            key: Cache key (usually file path)
            image: Image data as numpy array

        Returns:
            True if cached successfully, False if image too large
        """
        # Calculate image size
        image_size = image.nbytes

        # Don't cache if image alone exceeds max memory
        if image_size > self.max_memory_bytes:
            logger.warning(
                f"Image {key} too large to cache ({image_size / 1024 / 1024:.1f}MB)"
            )
            return False

        with self._lock:
            # If key already exists, remove old entry
            if key in self._cache:
                old_size = self._cache[key]["size"]
                self.current_memory_bytes -= old_size
                del self._cache[key]

            # Evict old entries until we have space
            while (
                self.current_memory_bytes + image_size
            ) > self.max_memory_bytes and self._cache:
                oldest_key, oldest_value = self._cache.popitem(last=False)
                self.current_memory_bytes -= oldest_value["size"]
                logger.debug("Evicted %s from cache", oldest_key)

            # Add new entry
            self._cache[key] = {
                "data": image.copy(),  # Store a copy to prevent external modifications
                "size": image_size,
            }
            self.current_memory_bytes += image_size
            logger.debug(
                f"Cached {key} ({image_size / 1024 / 1024:.1f}MB), "
                f"total: {self.current_memory_bytes / 1024 / 1024:.1f}MB"
            )
            return True

    def remove(self, key: str) -> bool:
        """
        Remove image from cache.

        Args:
            key: Cache key

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                size = self._cache[key]["size"]
                self.current_memory_bytes -= size
                del self._cache[key]
                logger.debug("Removed %s from cache", key)
                return True
            return False

    def clear(self):
        """Clear all cached images."""
        with self._lock:
            self._cache.clear()
            self.current_memory_bytes = 0
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared")

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            return {
                "size": len(self._cache),
                "memory_mb": self.current_memory_bytes / 1024 / 1024,
                "max_memory_mb": self.max_memory_bytes / 1024 / 1024,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def set_max_memory(self, max_memory_mb: int):
        """
        Update maximum memory limit.

        Args:
            max_memory_mb: New maximum memory in megabytes
        """
        with self._lock:
            self.max_memory_bytes = max_memory_mb * 1024 * 1024
            # Evict entries if we now exceed the limit
            while self.current_memory_bytes > self.max_memory_bytes and self._cache:
                oldest_key, oldest_value = self._cache.popitem(last=False)
                self.current_memory_bytes -= oldest_value["size"]
                logger.debug("Evicted %s due to new memory limit", oldest_key)
            logger.info("Cache memory limit updated to %dMB", max_memory_mb)

    def __len__(self):
        """Return number of cached images."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key):
        """Check if key is in cache."""
        with self._lock:
            return key in self._cache
