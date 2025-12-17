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

### Build Errors

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
