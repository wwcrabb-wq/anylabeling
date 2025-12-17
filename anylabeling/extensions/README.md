# Cython Extensions (Optional)

This directory contains optional Cython extensions for performance-critical operations in AnyLabeling.

## Status: Not Yet Implemented

The Cython extensions are **planned but not yet implemented**. The current implementation uses optimized Python code that provides good performance on all platforms without requiring compilation.

## Planned Extensions

When implemented, these extensions will provide:

### 1. fast_nms.pyx
- Optimized Non-Maximum Suppression (NMS) algorithm
- Expected speedup: 10-50x over pure Python
- C-level implementation with typed memoryviews

### 2. fast_transforms.pyx
- Fast coordinate transformations
- In-place operations to minimize memory allocation
- Letterbox transformation for model input

### 3. polygon_ops.pyx
- Polygon area calculation (Shoelace formula)
- Point-in-polygon tests
- Douglas-Peucker polygon simplification

## Building Extensions (When Implemented)

### Requirements
```bash
pip install cython>=3.0.0 numpy
```

### Build
```bash
cd anylabeling/extensions
python setup_extensions.py build_ext --inplace
```

### Platform-specific Notes

**Linux:**
```bash
# Install build tools
sudo apt-get install build-essential python3-dev
```

**macOS:**
```bash
# Install Xcode command line tools
xcode-select --install
```

**Windows:**
```bash
# Install Microsoft Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
```

## Fallback Behavior

AnyLabeling will automatically detect if extensions are available and fall back to Python implementations if not. This ensures the application works on all platforms without requiring compilation.

## Contributing

If you'd like to contribute these extensions:

1. Implement the .pyx files following the specifications in the main issue
2. Create Python fallback implementations
3. Add proper error handling and logging
4. Include benchmarks comparing Cython vs Python performance
5. Test on Linux, macOS, and Windows
6. Update this README with build instructions

## Performance Expectations

Based on similar projects, we expect:

| Operation | Python | Cython | Speedup |
|-----------|--------|--------|---------|
| NMS (1000 boxes) | ~50ms | ~2ms | 25x |
| Coordinate Transform | ~10ms | ~0.5ms | 20x |
| Polygon Area | ~5ms | ~0.2ms | 25x |

These are estimates and actual performance will vary based on hardware and input size.
