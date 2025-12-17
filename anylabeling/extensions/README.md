# AnyLabeling Extensions

This directory contains optional Cython extensions for performance-critical operations.

## Overview

The extensions provide significant speedups (10-50x) for:
- **fast_nms**: Non-Maximum Suppression (NMS) for object detection
- **fast_transforms**: Coordinate transformations and image normalization
- **polygon_ops**: Polygon operations (area, point-in-polygon, simplification, IoU)

## Building Extensions

### Prerequisites
- Python 3.8+
- Cython >= 3.0.0
- NumPy
- C compiler (GCC, Clang, or MSVC)

### Installation

```bash
# Install build dependencies
pip install cython numpy

# Build extensions
cd /path/to/anylabeling
python anylabeling/extensions/setup_extensions.py build_ext --inplace
```

### Verification

```python
from anylabeling.extensions import extensions_available, get_extension_status

print(f"Extensions available: {extensions_available()}")
print(f"Extension status: {get_extension_status()}")
```

## Fallback Behavior

If extensions are not built or fail to import, the module automatically falls back to pure Python implementations. The API remains identical, ensuring backward compatibility.

## Performance

Expected speedups with Cython extensions:
- NMS: 10-50x faster (varies with box count)
- Transforms: 5-15x faster
- Polygon operations: 10-30x faster

## Platform Notes

### Linux
- Use GCC or Clang compiler
- Best performance with `-march=native` flag

### Windows
- Requires Visual Studio Build Tools or MinGW
- Use `/O2` optimization flag

### macOS
- Use Clang (included with Xcode Command Line Tools)
- Apple Silicon: ensure NumPy is compiled for ARM64
