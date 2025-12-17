use pyo3::prelude::*;
use rayon::prelude::*;
use walkdir::WalkDir;

/// Scan directory for image files with parallel traversal
#[pyfunction]
#[pyo3(signature = (path, extensions=None, recursive=true))]
pub fn scan_image_directory(
    path: &str,
    extensions: Option<Vec<String>>,
    recursive: bool,
) -> PyResult<Vec<String>> {
    // Default image extensions
    let default_exts = vec![
        "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"
    ].into_iter().map(String::from).collect();
    
    let valid_extensions = extensions.unwrap_or(default_exts);
    
    // Set up walker
    let walker = if recursive {
        WalkDir::new(path).follow_links(true)
    } else {
        WalkDir::new(path).max_depth(1)
    };

    // Collect entries
    let entries: Vec<_> = walker
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .collect();

    // Filter by extension in parallel
    let image_paths: Vec<String> = entries
        .par_iter()
        .filter_map(|entry| {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if valid_extensions.iter().any(|e| e.to_lowercase() == ext_str) {
                    return path.to_string_lossy().to_string().into();
                }
            }
            None
        })
        .collect();

    // Sort for consistency
    let mut sorted_paths = image_paths;
    sorted_paths.sort();

    Ok(sorted_paths)
}
