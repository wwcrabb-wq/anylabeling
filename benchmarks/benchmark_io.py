"""
Benchmark image I/O operations.

Usage:
    python benchmarks/benchmark_io.py --images path/to/images/ --output results.json
"""

import argparse
import time
import numpy as np
from pathlib import Path
from typing import List, Dict
import json
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_sequential_loading(image_paths: List[str], loader_name: str = "opencv") -> Dict:
    """Benchmark sequential image loading."""
    logger.info(f"Benchmarking sequential loading with {loader_name}...")
    
    import cv2
    
    start = time.perf_counter()
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    end = time.perf_counter()
    
    elapsed = end - start
    throughput = len(images) / elapsed if elapsed > 0 else 0
    
    return {
        "loader": loader_name,
        "mode": "sequential",
        "count": len(images),
        "elapsed_s": elapsed,
        "throughput_images_per_sec": throughput,
    }


def benchmark_parallel_loading(image_paths: List[str], num_threads: int = 4) -> Dict:
    """Benchmark parallel image loading."""
    logger.info(f"Benchmarking parallel loading with {num_threads} threads...")
    
    import cv2
    
    def load_image(path):
        return cv2.imread(path)
    
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        images = list(executor.map(load_image, image_paths))
    end = time.perf_counter()
    
    images = [img for img in images if img is not None]
    elapsed = end - start
    throughput = len(images) / elapsed if elapsed > 0 else 0
    
    return {
        "loader": "opencv",
        "mode": f"parallel_{num_threads}threads",
        "count": len(images),
        "elapsed_s": elapsed,
        "throughput_images_per_sec": throughput,
    }


def benchmark_rust_loading(image_paths: List[str]) -> Dict:
    """Benchmark Rust extension image loading."""
    logger.info("Benchmarking Rust extension loading...")
    
    try:
        from anylabeling.rust_extensions import load_images_parallel, rust_available
        
        if not rust_available():
            return {"loader": "rust", "status": "unavailable"}
        
        start = time.perf_counter()
        images = load_images_parallel(image_paths)
        end = time.perf_counter()
        
        images = [img for img in images if img is not None]
        elapsed = end - start
        throughput = len(images) / elapsed if elapsed > 0 else 0
        
        return {
            "loader": "rust",
            "mode": "parallel",
            "count": len(images),
            "elapsed_s": elapsed,
            "throughput_images_per_sec": throughput,
        }
    except Exception as e:
        logger.warning(f"Rust loading failed: {e}")
        return {"loader": "rust", "status": "error", "error": str(e)}


def benchmark_directory_scanning(directory: str) -> Dict[str, Dict]:
    """Benchmark directory scanning methods."""
    logger.info(f"Benchmarking directory scanning for {directory}...")
    
    results = {}
    
    # Python pathlib
    start = time.perf_counter()
    py_paths = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        py_paths.extend(list(Path(directory).rglob(f"*{ext}")))
    elapsed_py = time.perf_counter() - start
    
    results["pathlib"] = {
        "count": len(py_paths),
        "elapsed_s": elapsed_py,
    }
    
    # Rust extension
    try:
        from anylabeling.rust_extensions import scan_image_directory, rust_available
        
        if rust_available():
            start = time.perf_counter()
            rust_paths = scan_image_directory(directory, recursive=True)
            elapsed_rust = time.perf_counter() - start
            
            results["rust"] = {
                "count": len(rust_paths),
                "elapsed_s": elapsed_rust,
                "speedup": elapsed_py / elapsed_rust if elapsed_rust > 0 else 0,
            }
    except Exception as e:
        logger.warning(f"Rust scanning failed: {e}")
        results["rust"] = {"status": "error", "error": str(e)}
    
    return results


def generate_report(results: Dict, output_path: str):
    """Generate I/O benchmark report."""
    output_path = Path(output_path)
    
    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Report saved to {output_path}")
    
    elif output_path.suffix == ".md":
        with open(output_path, "w") as f:
            f.write("# I/O Performance Report\n\n")
            
            if "loading" in results:
                f.write("## Image Loading\n\n")
                f.write("| Loader | Mode | Count | Time (s) | Throughput (img/s) |\n")
                f.write("|--------|------|-------|----------|--------------------|\n")
                for result in results["loading"]:
                    if "throughput_images_per_sec" in result:
                        f.write(f"| {result['loader']} | {result['mode']} | "
                               f"{result['count']} | {result['elapsed_s']:.3f} | "
                               f"{result['throughput_images_per_sec']:.2f} |\n")
                f.write("\n")
            
            if "scanning" in results:
                f.write("## Directory Scanning\n\n")
                for method, data in results["scanning"].items():
                    if "count" in data:
                        f.write(f"- **{method}**: {data['count']} files in {data['elapsed_s']:.3f}s\n")
                        if "speedup" in data:
                            f.write(f"  - Speedup: {data['speedup']:.2f}x\n")
        
        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark I/O operations")
    parser.add_argument("--images", type=str, required=True, help="Path to test images directory")
    parser.add_argument("--output", type=str, default="io_benchmark_results.json", help="Output file path")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of images to test")
    
    args = parser.parse_args()
    
    # Find image files
    image_dir = Path(args.images)
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend([str(p) for p in image_dir.rglob(ext)])
    
    if args.limit:
        image_paths = image_paths[:args.limit]
    
    logger.info(f"Found {len(image_paths)} images")
    
    results = {
        "directory": args.images,
        "image_count": len(image_paths),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loading": [],
        "scanning": {}
    }
    
    # Benchmark loading
    results["loading"].append(benchmark_sequential_loading(image_paths))
    results["loading"].append(benchmark_parallel_loading(image_paths, num_threads=4))
    results["loading"].append(benchmark_rust_loading(image_paths))
    
    # Benchmark scanning
    results["scanning"] = benchmark_directory_scanning(args.images)
    
    generate_report(results, args.output)


if __name__ == "__main__":
    main()
