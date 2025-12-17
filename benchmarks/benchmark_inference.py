"""
Benchmark model inference performance.
Measures single image, batch, and throughput metrics.

Usage:
    python benchmarks/benchmark_inference.py --model path/to/model.onnx --images path/to/images/
"""

import argparse
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_model(model, sample_input, iterations=10):
    """
    Warm up model before benchmarking.
    
    Args:
        model: Model instance
        sample_input: Sample input for model
        iterations: Number of warmup iterations
    """
    logger.info(f"Warming up model with {iterations} iterations...")
    for _ in range(iterations):
        _ = model.predict(sample_input)


def benchmark_single_image(model, images: List[np.ndarray], iterations=100) -> Dict[str, float]:
    """
    Benchmark single image inference with statistics.
    
    Args:
        model: Model instance
        images: List of test images
        iterations: Number of iterations
    
    Returns:
        Dictionary with timing statistics
    """
    logger.info(f"Benchmarking single image inference ({iterations} iterations)...")
    
    timings = []
    sample_image = images[0]
    
    # Warmup
    warmup_model(model, sample_image, 10)
    
    # Benchmark
    for i in range(iterations):
        start = time.perf_counter()
        _ = model.predict(sample_image)
        end = time.perf_counter()
        timings.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_ms": np.mean(timings),
        "std_ms": np.std(timings),
        "min_ms": np.min(timings),
        "max_ms": np.max(timings),
        "median_ms": np.median(timings),
        "p95_ms": np.percentile(timings, 95),
        "p99_ms": np.percentile(timings, 99),
    }


def benchmark_batch(model, images: List[np.ndarray], batch_sizes=[1, 2, 4, 8, 16]) -> Dict[int, Dict[str, float]]:
    """
    Benchmark batch inference at various sizes.
    
    Args:
        model: Model instance with batch support
        images: List of test images
        batch_sizes: List of batch sizes to test
    
    Returns:
        Dictionary mapping batch size to timing statistics
    """
    logger.info(f"Benchmarking batch inference (sizes: {batch_sizes})...")
    
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(images):
            logger.warning(f"Skipping batch size {batch_size} (not enough images)")
            continue
        
        batch = images[:batch_size]
        timings = []
        
        # Warmup
        for _ in range(5):
            _ = model.predict_batch(batch)
        
        # Benchmark
        for _ in range(50):
            start = time.perf_counter()
            _ = model.predict_batch(batch)
            end = time.perf_counter()
            timings.append((end - start) * 1000)
        
        results[batch_size] = {
            "mean_ms": np.mean(timings),
            "std_ms": np.std(timings),
            "throughput_images_per_sec": (batch_size * 1000) / np.mean(timings),
        }
    
    return results


def benchmark_backends(onnx_path: str, images: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compare performance across all available backends.
    
    Args:
        onnx_path: Path to ONNX model
        images: List of test images
    
    Returns:
        Dictionary mapping backend name to performance metrics
    """
    logger.info("Benchmarking available backends...")
    
    results = {}
    
    # Try ONNX Runtime CPU
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        logger.info("Testing ONNX Runtime CPU...")
        # Simplified benchmark - would need proper model wrapper
        results["onnx_cpu"] = {"status": "available"}
    except Exception as e:
        logger.warning(f"ONNX CPU not available: {e}")
        results["onnx_cpu"] = {"status": "unavailable", "error": str(e)}
    
    # Try ONNX Runtime GPU
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
            logger.info("Testing ONNX Runtime GPU...")
            results["onnx_gpu"] = {"status": "available"}
        else:
            results["onnx_gpu"] = {"status": "unavailable", "error": "CUDA not available"}
    except Exception as e:
        logger.warning(f"ONNX GPU not available: {e}")
        results["onnx_gpu"] = {"status": "unavailable", "error": str(e)}
    
    # Try TensorRT
    try:
        from anylabeling.services.auto_labeling.tensorrt_backend import TENSORRT_AVAILABLE
        if TENSORRT_AVAILABLE:
            logger.info("Testing TensorRT...")
            results["tensorrt"] = {"status": "available"}
        else:
            results["tensorrt"] = {"status": "unavailable", "error": "TensorRT not installed"}
    except Exception as e:
        logger.warning(f"TensorRT not available: {e}")
        results["tensorrt"] = {"status": "unavailable", "error": str(e)}
    
    return results


def generate_report(results: Dict[str, Any], output_path: str):
    """
    Generate performance report.
    
    Args:
        results: Benchmark results
        output_path: Path to output file (JSON or Markdown)
    """
    output_path = Path(output_path)
    
    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Report saved to {output_path}")
    
    elif output_path.suffix == ".md":
        with open(output_path, "w") as f:
            f.write("# Inference Performance Report\n\n")
            
            if "single_image" in results:
                f.write("## Single Image Inference\n\n")
                for key, value in results["single_image"].items():
                    f.write(f"- **{key}**: {value:.2f}\n")
                f.write("\n")
            
            if "batch" in results:
                f.write("## Batch Inference\n\n")
                f.write("| Batch Size | Mean (ms) | Std (ms) | Throughput (img/s) |\n")
                f.write("|------------|-----------|----------|--------------------|\n")
                for batch_size, metrics in results["batch"].items():
                    f.write(f"| {batch_size} | {metrics['mean_ms']:.2f} | "
                           f"{metrics['std_ms']:.2f} | {metrics['throughput_images_per_sec']:.2f} |\n")
                f.write("\n")
            
            if "backends" in results:
                f.write("## Backend Availability\n\n")
                for backend, info in results["backends"].items():
                    status = info.get("status", "unknown")
                    f.write(f"- **{backend}**: {status}\n")
        
        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference performance")
    parser.add_argument("--model", type=str, help="Path to ONNX model")
    parser.add_argument("--images", type=str, help="Path to test images directory")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file path")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    
    args = parser.parse_args()
    
    # This is a simplified implementation
    # Full implementation would load actual model and images
    logger.info("Inference benchmarking suite")
    logger.info(f"Model: {args.model}")
    logger.info(f"Images: {args.images}")
    
    results = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Full implementation requires model loading"
    }
    
    generate_report(results, args.output)


if __name__ == "__main__":
    main()
