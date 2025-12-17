use pyo3::prelude::*;

mod image_loader;
mod directory_scanner;
mod mmap_reader;

use image_loader::load_images_parallel;
use directory_scanner::scan_image_directory;
use mmap_reader::MmapImageReader;

/// High-performance Rust extensions for AnyLabeling
#[pymodule]
fn anylabeling_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_images_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(scan_image_directory, m)?)?;
    m.add_class::<MmapImageReader>()?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
