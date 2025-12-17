"""Parallel processing utilities for image loading and batch operations."""

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Any
from multiprocessing import cpu_count

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ParallelImageLoader:
    """Parallel image loader using ThreadPoolExecutor for async image loading."""

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel image loader.

        Args:
            max_workers: Maximum number of worker threads. Defaults to min(8, cpu_count())
        """
        if max_workers is None:
            max_workers = min(8, cpu_count())
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"ParallelImageLoader initialized with {max_workers} workers")

    def load_images(
        self, image_paths: List[str], convert_to_rgb: bool = True
    ) -> List[Optional[np.ndarray]]:
        """
        Load multiple images in parallel.

        Args:
            image_paths: List of image file paths
            convert_to_rgb: Convert images to RGB mode

        Returns:
            List of numpy arrays (or None if loading failed)
        """

        def load_single_image(path: str) -> Optional[np.ndarray]:
            try:
                img = Image.open(path)
                if convert_to_rgb and img.mode != "RGB":
                    img = img.convert("RGB")
                return np.array(img)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                return None

        # Submit all tasks
        futures = {
            self.executor.submit(load_single_image, path): idx
            for idx, path in enumerate(image_paths)
        }

        # Collect results in order
        results = [None] * len(image_paths)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Error loading image at index {idx}: {e}")
                results[idx] = None

        return results

    def load_images_async(
        self,
        image_paths: List[str],
        callback: Optional[Callable[[int, Optional[np.ndarray]], None]] = None,
        convert_to_rgb: bool = True,
    ):
        """
        Load images asynchronously with callback for each loaded image.

        Args:
            image_paths: List of image file paths
            callback: Optional callback function called with (index, image) when each image is loaded
            convert_to_rgb: Convert images to RGB mode
        """

        def load_single_image(idx: int, path: str):
            try:
                img = Image.open(path)
                if convert_to_rgb and img.mode != "RGB":
                    img = img.convert("RGB")
                result = np.array(img)
                if callback:
                    callback(idx, result)
                return result
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                if callback:
                    callback(idx, None)
                return None

        # Submit all tasks
        for idx, path in enumerate(image_paths):
            self.executor.submit(load_single_image, idx, path)

    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class BatchProcessor:
    """Process items in parallel batches with thread-safe queue management."""

    def __init__(
        self,
        process_func: Callable[[Any], Any],
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize batch processor.

        Args:
            process_func: Function to process each item
            max_workers: Maximum number of worker threads
            batch_size: Size of each batch (not used for threading, for compatibility)
        """
        if max_workers is None:
            max_workers = min(8, cpu_count())
        self.max_workers = max_workers
        self.process_func = process_func
        self.batch_size = batch_size or 1
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._results_queue = queue.Queue()
        self._lock = threading.Lock()
        logger.info(f"BatchProcessor initialized with {max_workers} workers")

    def process_items(
        self,
        items: List[Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """
        Process items in parallel.

        Args:
            items: List of items to process
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of processed results in the same order as input
        """
        total = len(items)
        results = [None] * total
        completed = [0]  # Use list to make it mutable in closure

        def process_with_index(idx: int, item: Any) -> tuple:
            result = self.process_func(item)
            return idx, result

        # Submit all tasks
        futures = {
            self.executor.submit(process_with_index, idx, item): idx
            for idx, item in enumerate(items)
        }

        # Collect results
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0], total)
            except Exception as e:
                idx = futures[future]
                logger.error(f"Error processing item at index {idx}: {e}")
                results[idx] = None
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0], total)

        return results

    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
