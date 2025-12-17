use pyo3::prelude::*;
use memmap2::Mmap;
use std::fs::File;

/// Memory-mapped image reader for large files
#[pyclass]
pub struct MmapImageReader {
    mmap: Mmap,
    path: String,
}

#[pymethods]
impl MmapImageReader {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to open file {}: {}", path, e)
            ))?;
        
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    format!("Failed to memory map file: {}", e)
                ))?
        };

        Ok(MmapImageReader {
            mmap,
            path: path.to_string(),
        })
    }

    pub fn read_bytes(&self, offset: usize, length: usize) -> PyResult<Vec<u8>> {
        if offset + length > self.mmap.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Read beyond file bounds"
            ));
        }
        
        Ok(self.mmap[offset..offset + length].to_vec())
    }

    pub fn get_size(&self) -> usize {
        self.mmap.len()
    }

    pub fn get_path(&self) -> &str {
        &self.path
    }
}
