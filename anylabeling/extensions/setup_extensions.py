"""
Build script for Cython extensions.

Usage:
    python anylabeling/extensions/setup_extensions.py build_ext --inplace

This will compile the Cython extensions and place them in the extensions directory.
"""

from setuptools import setup, Extension
import sys
import os

try:
    from Cython.Build import cythonize
    import numpy as np
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("Warning: Cython or numpy not available. Extensions cannot be built.")
    print("Install with: pip install cython numpy")

if not CYTHON_AVAILABLE:
    sys.exit(1)

# Extension definitions
extensions = [
    Extension(
        "anylabeling.extensions.fast_nms",
        ["anylabeling/extensions/fast_nms.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"] if sys.platform != "win32" else ["/O2"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "anylabeling.extensions.fast_transforms",
        ["anylabeling/extensions/fast_transforms.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"] if sys.platform != "win32" else ["/O2"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "anylabeling.extensions.polygon_ops",
        ["anylabeling/extensions/polygon_ops.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"] if sys.platform != "win32" else ["/O2"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

# Compiler directives for optimization
compiler_directives = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
    "nonecheck": False,
}

if __name__ == "__main__":
    setup(
        name="anylabeling_extensions",
        ext_modules=cythonize(
            extensions,
            compiler_directives=compiler_directives,
            annotate=True,  # Generate HTML annotation files for optimization analysis
        ),
        zip_safe=False,
    )
