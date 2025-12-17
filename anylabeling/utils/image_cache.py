"""Image caching utilities with LRU eviction policy."""

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, List, Any

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


class FilterResultCache:
    """Disk-persistent cache for image filter results with LRU eviction."""

    def __init__(self, cache_dir: Optional[str] = None, max_entries: int = 100):
        """
        Initialize filter result cache.

        Args:
            cache_dir: Directory for cache files (default: ~/.anylabeling/filter_cache/)
            max_entries: Maximum number of cached results
        """
        if cache_dir is None:
            home_dir = Path.home()
            cache_dir = home_dir / ".anylabeling" / "filter_cache"
        
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cache index
        self.index_file = self.cache_dir / "index.json"
        self._load_index()
        
        logger.info("FilterResultCache initialized at %s", self.cache_dir)

    def _load_index(self):
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    data = json.load(f)
                    # Convert to OrderedDict to maintain LRU order
                    self._index = OrderedDict(data.get("entries", {}))
            except Exception as e:
                logger.warning("Failed to load cache index: %s", e)
                self._index = OrderedDict()
        else:
            self._index = OrderedDict()

    def _save_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_file, "w") as f:
                json.dump({"entries": dict(self._index)}, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save cache index: %s", e)

    def _generate_cache_key(
        self,
        folder_path: str,
        model_name: str,
        min_confidence: float,
        max_confidence: float,
        selected_classes: Optional[List[str]],
        count_mode: str,
        count_value: int,
    ) -> str:
        """
        Generate a unique cache key for filter parameters.

        Args:
            folder_path: Path to image folder
            model_name: Name of the model
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            selected_classes: List of selected class names (None = all)
            count_mode: Detection count mode
            count_value: Detection count value

        Returns:
            Cache key string
        """
        # Create a string representation of parameters
        key_parts = [
            folder_path,
            model_name,
            f"{min_confidence:.2f}",
            f"{max_confidence:.2f}",
            str(sorted(selected_classes) if selected_classes else "all"),
            count_mode,
            str(count_value),
        ]
        key_string = "|".join(key_parts)
        
        # Generate hash for the key
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        return cache_key

    def get(
        self,
        folder_path: str,
        model_name: str,
        min_confidence: float,
        max_confidence: float,
        selected_classes: Optional[List[str]],
        count_mode: str,
        count_value: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached filter results.

        Returns:
            Dictionary with 'filtered_images' list and metadata, or None if not cached
        """
        cache_key = self._generate_cache_key(
            folder_path,
            model_name,
            min_confidence,
            max_confidence,
            selected_classes,
            count_mode,
            count_value,
        )

        with self._lock:
            if cache_key not in self._index:
                self._misses += 1
                return None

            # Check if cache file exists
            cache_file = self.cache_dir / f"{cache_key}.json"
            if not cache_file.exists():
                # Remove stale index entry
                del self._index[cache_key]
                self._save_index()
                self._misses += 1
                return None

            # Check if folder has been modified since cache was created
            cache_mtime = self._index[cache_key].get("mtime", 0)
            try:
                folder_mtime = os.path.getmtime(folder_path)
                if folder_mtime > cache_mtime:
                    # Folder modified, invalidate cache
                    logger.info("Cache invalidated due to folder modification")
                    cache_file.unlink()
                    del self._index[cache_key]
                    self._save_index()
                    self._misses += 1
                    return None
            except OSError:
                pass  # Folder might not exist anymore

            # Load cache file
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                
                # Move to end (most recently used)
                self._index.move_to_end(cache_key)
                self._save_index()
                
                self._hits += 1
                logger.info("Cache hit for key %s", cache_key[:8])
                return data
            except Exception as e:
                logger.warning("Failed to load cache file %s: %s", cache_file, e)
                self._misses += 1
                return None

    def put(
        self,
        folder_path: str,
        model_name: str,
        min_confidence: float,
        max_confidence: float,
        selected_classes: Optional[List[str]],
        count_mode: str,
        count_value: int,
        filtered_images: List[str],
        total_images: int,
    ):
        """
        Cache filter results.

        Args:
            folder_path: Path to image folder
            model_name: Name of the model
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            selected_classes: List of selected class names (None = all)
            count_mode: Detection count mode
            count_value: Detection count value
            filtered_images: List of filtered image paths
            total_images: Total number of images processed
        """
        cache_key = self._generate_cache_key(
            folder_path,
            model_name,
            min_confidence,
            max_confidence,
            selected_classes,
            count_mode,
            count_value,
        )

        with self._lock:
            # Evict old entries if we're at capacity
            while len(self._index) >= self.max_entries:
                oldest_key, oldest_entry = self._index.popitem(last=False)
                oldest_file = self.cache_dir / f"{oldest_key}.json"
                if oldest_file.exists():
                    oldest_file.unlink()
                logger.debug("Evicted cache entry %s", oldest_key[:8])

            # Save cache file
            cache_file = self.cache_dir / f"{cache_key}.json"
            try:
                folder_mtime = os.path.getmtime(folder_path)
            except OSError:
                folder_mtime = time.time()

            data = {
                "filtered_images": filtered_images,
                "total_images": total_images,
                "filter_params": {
                    "folder_path": folder_path,
                    "model_name": model_name,
                    "min_confidence": min_confidence,
                    "max_confidence": max_confidence,
                    "selected_classes": selected_classes,
                    "count_mode": count_mode,
                    "count_value": count_value,
                },
                "timestamp": time.time(),
            }

            try:
                with open(cache_file, "w") as f:
                    json.dump(data, f, indent=2)

                # Update index
                self._index[cache_key] = {
                    "mtime": folder_mtime,
                    "timestamp": time.time(),
                }
                self._save_index()

                logger.info("Cached filter results with key %s", cache_key[:8])
            except Exception as e:
                logger.warning("Failed to save cache file: %s", e)

    def clear(self):
        """Clear all cached results."""
        with self._lock:
            # Delete all cache files
            for cache_key in list(self._index.keys()):
                cache_file = self.cache_dir / f"{cache_key}.json"
                if cache_file.exists():
                    cache_file.unlink()

            # Clear index
            self._index.clear()
            self._save_index()

            # Reset stats
            self._hits = 0
            self._misses = 0

            logger.info("Filter result cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            # Calculate cache size on disk
            cache_size_bytes = 0
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_size_bytes += cache_file.stat().st_size
                except OSError:
                    pass

            return {
                "entries": len(self._index),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size_mb": cache_size_bytes / 1024 / 1024,
                "cache_dir": str(self.cache_dir),
            }
