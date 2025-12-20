# Rust Extensions for AnyLabeling

High-performance I/O operations using Rust.

## Overview

Rust extensions provide 5-10x speedup for:
- **Parallel image loading**: Load multiple images concurrently using Rayon
- **Directory scanning**: Fast recursive directory traversal with WalkDir
- **Memory-mapped I/O**: Zero-copy file reading for large files

## Building

### Prerequisites
- Rust toolchain (install from https://rustup.rs/)
- Python 3.8+
- Maturin: `pip install maturin`

### Build for Development
```bash
cd anylabeling/rust_extensions
maturin develop --release
```

### Build Wheel
```bash
cd anylabeling/rust_extensions
maturin build --release
```

### Install
```bash
pip install target/wheels/anylabeling_rust-*.whl
```

## Usage

```python
from anylabeling.rust_extensions import (
    load_images_parallel,
    scan_image_directory,
    MmapImageReader,
    rust_available
)

# Check availability
if rust_available():
    print("Rust extensions are available")
else:
    print("Using Python fallback")

# Scan directory for images
image_paths = scan_image_directory(
    "/path/to/images",
    extensions=["jpg", "png"],
    recursive=True
)

# Load images in parallel
images = load_images_parallel(image_paths, num_threads=4)

# Memory-mapped reading
reader = MmapImageReader("/path/to/large/file.bin")
data = reader.read_bytes(offset=0, length=1024)
```

## Performance

Expected speedups with Rust extensions:
- Directory scanning: 5-10x faster than Python glob
- Parallel image loading: 3-5x faster than ThreadPoolExecutor
- Memory-mapped I/O: Minimal overhead for large files

## Fallback Behavior

If Rust extensions are not built or fail to import, the module automatically falls back to pure Python implementations using ThreadPoolExecutor and pathlib. The API remains identical.

## Platform Support

- **Linux**: Full support (tested on Ubuntu, Debian, CentOS)
- **Windows**: Full support (requires MSVC or MinGW)
- **macOS**: Full support (Intel and Apple Silicon)

## Troubleshooting

### Automated Build (Windows Batch Script)

The `start_anylabeling.bat` script now includes enhanced error handling for Rust extensions:

**Checks performed before building**:
1. Verifies Rust toolchain is installed
2. Checks for required files (Cargo.toml, lib.rs)
3. Detects available linkers (link.exe or lld.exe)
4. Displays detailed error messages with actionable steps

**If build fails**, the script will:
- Display the full build output for debugging
- Provide specific troubleshooting steps
- Continue with Python fallback implementations

**Common error messages and solutions**:

**Warning**: `No linker detected in PATH (link.exe or lld.exe)`
- Install "Desktop development with C++" from Visual Studio, or
- Install "Build Tools for Visual Studio" from https://visualstudio.microsoft.com/downloads/

**Warning**: `Cargo.toml not found in anylabeling\rust_extensions\`
- Ensure you have the complete source code (not just the Python package)
- Re-clone the repository if files are missing

**Warning**: `lib.rs not found in anylabeling\rust_extensions\src\`
- Ensure you have the complete source code
- Check that the repository is not corrupted

### Manual Build Errors

**Problem**: `error: linker 'cc' not found`
**Solution**: Install build tools
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

**Problem**: `error: could not find 'Cargo.toml'`
**Solution**: Make sure you're in the `anylabeling/rust_extensions` directory

**Problem**: `Couldn't find a virtualenv or conda environment`
**Solution**: Activate your virtual environment before running `maturin develop`:
```bash
# Windows
venv\Scripts\activate.bat

# Linux/macOS
source venv/bin/activate
```

**Problem**: `ImportError: cannot import name 'anylabeling_rust'`
**Solution**: Rebuild extensions with `maturin develop --release`

## Testing

```bash
pytest tests/test_rust_extensions.py
```

## Contributing

Contributions are welcome! Please:
1. Test on multiple platforms
2. Maintain Python fallback compatibility
3. Add benchmarks for performance claims
4. Update documentation
