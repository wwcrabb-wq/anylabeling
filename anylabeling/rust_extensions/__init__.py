"""
Rust extensions module with automatic fallback to Python implementations.

To build Rust extensions:
    pip install maturin
    cd anylabeling/rust_extensions
    maturin develop --release
"""

import logging

logger = logging.getLogger(__name__)

# Availability flag
RUST_AVAILABLE = False

# Try importing Rust extensions
try:
    from .anylabeling_rust import (
        load_images_parallel,
        scan_image_directory,
        MmapImageReader,
    )
    RUST_AVAILABLE = True
    logger.info("Using Rust I/O implementations")
except ImportError as e:
    logger.info(f"Rust extensions not available, using Python fallback: {e}")
    from .fallback import (
        load_images_parallel_python as load_images_parallel,
        scan_image_directory_python as scan_image_directory,
        MmapImageReaderPython as MmapImageReader,
    )


def rust_available():
    """Check if Rust extensions are available."""
    return RUST_AVAILABLE


__all__ = [
    "load_images_parallel",
    "scan_image_directory",
    "MmapImageReader",
    "rust_available",
    "RUST_AVAILABLE",
]
