"""
Pure Python fallback implementations for Rust extensions.
Used when Rust extensions are not available or fail to import.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_images_parallel_python(
    paths: List[str],
    num_threads: Optional[int] = None
) -> List[Optional[np.ndarray]]:
    """
    Load multiple images in parallel using ThreadPoolExecutor (Python fallback).
    
    Args:
        paths: List of image paths to load
        num_threads: Number of worker threads (None for default)
    
    Returns:
        List of numpy arrays (None for failed loads)
    """
    import cv2
    
    def load_single_image(path):
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            return None
    
    max_workers = num_threads if num_threads is not None else min(8, len(paths))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(load_single_image, paths))
    
    return results


def scan_image_directory_python(
    path: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> List[str]:
    """
    Scan directory for image files (Python fallback).
    
    Args:
        path: Directory path to scan
        extensions: List of valid extensions (None for default)
        recursive: Whether to scan recursively
    
    Returns:
        Sorted list of image file paths
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"]
    else:
        extensions = ["." + ext.lower().lstrip(".") for ext in extensions]
    
    image_paths = []
    
    path_obj = Path(path)
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in path_obj.glob(pattern):
        if file_path.is_file():
            if any(file_path.suffix.lower() == ext for ext in extensions):
                image_paths.append(str(file_path.absolute()))
    
    return sorted(image_paths)


class MmapImageReaderPython:
    """
    Memory-mapped image reader (Python fallback).
    
    Uses simple file I/O since Python's mmap is less performant.
    """
    
    def __init__(self, path: str):
        """
        Initialize reader.
        
        Args:
            path: Path to file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        self.path = path
        self._size = os.path.getsize(path)
    
    def read_bytes(self, offset: int, length: int) -> bytes:
        """
        Read bytes from file.
        
        Args:
            offset: Byte offset
            length: Number of bytes to read
        
        Returns:
            Bytes read from file
        """
        if offset + length > self._size:
            raise IndexError("Read beyond file bounds")
        
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(length)
    
    def get_size(self) -> int:
        """Get file size in bytes."""
        return self._size
    
    def get_path(self) -> str:
        """Get file path."""
        return self.path
