"""
Optional Cython extensions for performance-critical operations.

These extensions provide significant speedups for NMS, coordinate transforms,
and polygon operations. They are optional and will fall back to Python
implementations if not available.

To build the extensions:
    python anylabeling/extensions/setup_extensions.py build_ext --inplace

Requirements:
    - Cython>=3.0.0
    - numpy
    - C compiler (gcc, clang, or MSVC)
"""

import logging

logger = logging.getLogger(__name__)

# Try importing compiled extensions
_extensions_available = False

try:
    # When compiled, these would be imported here
    # from . import fast_nms, fast_transforms, polygon_ops
    _extensions_available = False  # Set to True when extensions are built
except ImportError:
    logger.debug("Cython extensions not available, using Python fallbacks")
    _extensions_available = False


def extensions_available():
    """Check if Cython extensions are available."""
    return _extensions_available


# Export availability flag
__all__ = ["extensions_available"]
