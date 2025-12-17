"""
Extensions module with automatic fallback to Python implementations.

This module attempts to import compiled Cython extensions and falls back
to pure Python implementations if they are not available.

To build Cython extensions:
    python anylabeling/extensions/setup_extensions.py build_ext --inplace
"""

import logging

logger = logging.getLogger(__name__)

# Availability flags
CYTHON_NMS_AVAILABLE = False
CYTHON_TRANSFORMS_AVAILABLE = False
CYTHON_POLYGON_AVAILABLE = False


# Try importing fast_nms
try:
    from .fast_nms import fast_nms_cython as fast_nms
    CYTHON_NMS_AVAILABLE = True
    logger.info("Using Cython NMS implementation")
except ImportError:
    from .fallbacks import fast_nms_python as fast_nms
    logger.info("Cython NMS not available, using Python fallback")


# Try importing fast_transforms
try:
    from .fast_transforms import (
        transform_coordinates_inplace as transform_coordinates,
        normalize_image,
    )
    CYTHON_TRANSFORMS_AVAILABLE = True
    logger.info("Using Cython transform implementations")
except ImportError:
    from .fallbacks import (
        transform_coordinates_python as transform_coordinates,
        normalize_image_python as normalize_image,
    )
    logger.info("Cython transforms not available, using Python fallback")


# Try importing polygon_ops
try:
    from .polygon_ops import (
        polygon_area,
        point_in_polygon,
        simplify_polygon,
        polygon_iou,
    )
    CYTHON_POLYGON_AVAILABLE = True
    logger.info("Using Cython polygon implementations")
except ImportError:
    from .fallbacks import (
        polygon_area_python as polygon_area,
        point_in_polygon_python as point_in_polygon,
        simplify_polygon_python as simplify_polygon,
        polygon_iou_python as polygon_iou,
    )
    logger.info("Cython polygon ops not available, using Python fallback")


def extensions_available():
    """Check if any Cython extensions are available."""
    return any([
        CYTHON_NMS_AVAILABLE,
        CYTHON_TRANSFORMS_AVAILABLE,
        CYTHON_POLYGON_AVAILABLE,
    ])


def get_extension_status():
    """Get detailed status of all extensions."""
    return {
        "nms": "cython" if CYTHON_NMS_AVAILABLE else "python",
        "transforms": "cython" if CYTHON_TRANSFORMS_AVAILABLE else "python",
        "polygon_ops": "cython" if CYTHON_POLYGON_AVAILABLE else "python",
    }


# Export all functions
__all__ = [
    # Functions
    "fast_nms",
    "transform_coordinates",
    "normalize_image",
    "polygon_area",
    "point_in_polygon",
    "simplify_polygon",
    "polygon_iou",
    # Status functions
    "extensions_available",
    "get_extension_status",
    # Flags
    "CYTHON_NMS_AVAILABLE",
    "CYTHON_TRANSFORMS_AVAILABLE",
    "CYTHON_POLYGON_AVAILABLE",
]
