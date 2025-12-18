# Building Optional Performance Extensions

This guide provides detailed instructions for building optional performance extensions for AnyLabeling.

## Overview

AnyLabeling includes three types of optional extensions:

1. **Cython Extensions** - Compiled Python extensions (C)
2. **Rust Extensions** - Rust-based extensions via PyO3
3. **TensorRT** - NVIDIA GPU acceleration

All extensions are **optional** and AnyLabeling works perfectly without them. Build only the extensions relevant to your use case.

> **Note for Windows users:** The `start_anylabeling.bat` script automatically attempts to build Cython and Rust extensions on first run if they are not already present. You can use this guide for manual builds or troubleshooting.

---

## Cython Extensions

### Prerequisites

**All Platforms:**
- Python 3.8 or later
- pip
- NumPy

**Linux:**
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

**Windows:**
- Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/downloads/)
- Or install [MinGW-w64](https://www.mingw-w64.org/)

### Building

```bash
# Install build dependencies
pip install cython numpy

# Navigate to repository root
cd /path/to/anylabeling

# Build extensions
python anylabeling/extensions/setup_extensions.py build_ext --inplace
```

### Verification

```python
from anylabeling.extensions import extensions_available, get_extension_status

if extensions_available():
    print("✓ Cython extensions available")
    print(get_extension_status())
else:
    print("✗ Cython extensions not available (using Python fallback)")
```

### Troubleshooting

**Problem:** `error: Microsoft Visual C++ 14.0 or greater is required`
**Solution:** Install Visual Studio Build Tools on Windows

**Problem:** `fatal error: Python.h: No such file or directory`
**Solution:** Install python3-dev package:
```bash
sudo apt-get install python3-dev
```

**Problem:** Build succeeds but import fails
**Solution:** Ensure extensions are built in-place with `--inplace` flag

---

## Rust Extensions

### Prerequisites

**All Platforms:**
- Python 3.8 or later
- Rust toolchain (rustc, cargo)
- Maturin

### Installing Rust

**Linux/macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Windows:**
- Download and run [rustup-init.exe](https://rustup.rs/)
- Follow installer prompts
- Restart terminal

**Verify Installation:**
```bash
rustc --version
cargo --version
```

### Building

```bash
# Install Maturin
pip install maturin

# Navigate to Rust extensions directory
cd anylabeling/rust_extensions

# Build for development
maturin develop --release

# Or build wheel
maturin build --release
# Then: pip install target/wheels/anylabeling_rust-*.whl
```

### Verification

```python
from anylabeling.rust_extensions import rust_available

if rust_available():
    print("✓ Rust extensions available")
else:
    print("✗ Rust extensions not available (using Python fallback)")
```

### Troubleshooting

**Problem:** `error: linker 'cc' not found`
**Solution (Linux):** 
```bash
sudo apt-get install build-essential
```

**Problem:** `error: could not find 'Cargo.toml'`
**Solution:** Ensure you're in `anylabeling/rust_extensions` directory

**Problem:** ImportError after building
**Solution:** Run `maturin develop --release` again

**Problem:** Build fails on Windows
**Solution:** Ensure MSVC or MinGW is properly installed

---

## TensorRT (NVIDIA GPU Only)

### Prerequisites

- **NVIDIA GPU** with Compute Capability >= 5.0
- **CUDA Toolkit** >= 12.0
- **cuDNN** >= 8.6
- **Linux or Windows** (TensorRT not officially supported on macOS)

### Installation

#### Step 1: Install CUDA Toolkit

**Linux (Ubuntu/Debian):**
```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA
sudo apt-get install cuda-toolkit-12-0
```

**Windows:**
- Download CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
- Run installer
- Add CUDA to PATH

#### Step 2: Install TensorRT

**Option A: pip (Recommended)**
```bash
pip install nvidia-tensorrt
```

**Option B: tar.gz (Linux)**
```bash
# Download TensorRT from https://developer.nvidia.com/tensorrt
tar -xzvf TensorRT-8.6.x.Linux.x86_64-gnu.cuda-12.0.tar.gz
cd TensorRT-8.6.x
pip install python/tensorrt-*-cp3*.whl
```

#### Step 3: Install Additional Dependencies

```bash
pip install pycuda cupy-cuda12x
```

### Verification

```python
import tensorrt as trt
import pycuda.autoinit

print(f"TensorRT version: {trt.__version__}")
print(f"CUDA available: {pycuda.autoinit is not None}")
```

### Troubleshooting

**Problem:** `ImportError: libnvinfer.so.8: cannot open shared object file`
**Solution:** Add TensorRT lib to LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH
```

**Problem:** `cudaGetDevice() failed. Status: CUDA driver version is insufficient`
**Solution:** Update NVIDIA driver:
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

**Problem:** Engine build fails
**Solution:** Check CUDA memory availability and reduce workspace size

---

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

**Best Support:** All extensions work well on Linux

**Tips:**
- Use system package manager for compilers
- Check `python3-dev` is installed for Cython
- Use virtual environment to avoid permission issues

### macOS

**Cython:** Works on both Intel and Apple Silicon
**Rust:** Works on both architectures
**TensorRT:** Not supported (no NVIDIA GPUs)

**Apple Silicon Notes:**
- Ensure NumPy is compiled for ARM64
- Use `arch -arm64` prefix if needed
- Metal Performance Shaders (MPS) is alternative to CUDA (not yet implemented)

### Windows

**Cython:** Requires Visual Studio Build Tools or MinGW
**Rust:** Works well with MSVC
**TensorRT:** Supported with CUDA

**Tips:**
- Use PowerShell with admin rights
- Add Python and compilers to PATH
- Visual Studio 2019 or later recommended

---

## Performance Testing

After building extensions, run benchmarks to verify improvements:

```bash
# Test NMS performance
python benchmarks/benchmark_nms.py

# Test I/O performance
python benchmarks/benchmark_io.py --images /path/to/images/

# Run all benchmarks
python benchmarks/run_benchmarks.py --output-dir results/
```

Expected speedups:
- **Cython NMS:** 10-50x faster
- **Rust I/O:** 5-10x faster
- **TensorRT:** 2-5x faster (FP16 mode)

---

## Uninstalling Extensions

### Cython Extensions
```bash
# Remove compiled files
cd anylabeling/extensions
rm -f *.so *.pyd *.c
```

### Rust Extensions
```bash
pip uninstall anylabeling_rust
```

### TensorRT
```bash
pip uninstall nvidia-tensorrt pycuda cupy-cuda12x
```

---

## Getting Help

If you encounter issues:

1. Check this guide thoroughly
2. Review error messages carefully
3. Ensure all prerequisites are installed
4. Check GitHub issues for similar problems
5. Open a new issue with:
   - Operating system and version
   - Python version
   - Full error message
   - Steps to reproduce

---

## Contributing

Improvements to build process are welcome! When contributing:

- Test on multiple platforms
- Update this documentation
- Maintain backward compatibility
- Provide clear error messages
- Add verification steps
