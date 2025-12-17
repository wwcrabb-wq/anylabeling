use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use numpy::{PyArray3, IntoPyArray};

/// Load multiple images in parallel using Rayon thread pool
#[pyfunction]
#[pyo3(signature = (paths, num_threads=None))]
pub fn load_images_parallel(
    py: Python,
    paths: Vec<String>,
    num_threads: Option<usize>,
) -> PyResult<PyObject> {
    // Configure thread pool
    let pool = if let Some(n) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create thread pool: {}", e)))?
    } else {
        rayon::ThreadPoolBuilder::new()
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create thread pool: {}", e)))?
    };

    // Load images in parallel
    let results: Vec<Option<(Vec<u8>, (usize, usize, usize))>> = pool.install(|| {
        paths.par_iter().map(|path| {
            load_image_from_path(path)
        }).collect()
    });

    // Convert to Python list
    let py_list = PyList::empty(py);
    for result in results {
        match result {
            Some((data, (height, width, channels))) => {
                // Convert to numpy array
                let array = PyArray3::<u8>::from_vec3(
                    py,
                    &vec![
                        vec![
                            vec![0u8; channels]; width
                        ]; height
                    ],
                )?;
                
                // Fill array with data
                unsafe {
                    let slice = array.as_slice_mut()?;
                    slice.copy_from_slice(&data);
                }
                
                py_list.append(array)?;
            }
            None => {
                py_list.append(py.None())?;
            }
        }
    }

    Ok(py_list.into())
}

fn load_image_from_path(path: &str) -> Option<(Vec<u8>, (usize, usize, usize))> {
    use image::GenericImageView;

    let img = image::open(path).ok()?;
    let (width, height) = img.dimensions();
    let rgb_img = img.to_rgb8();
    let shape = (height as usize, width as usize, 3);
    
    Some((rgb_img.into_raw(), shape))
}
