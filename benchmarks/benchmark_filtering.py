#!/usr/bin/env python3
"""
Benchmark image filtering performance.

This script benchmarks the image filter dialog performance with different:
- Dataset sizes
- Sequential vs parallel filtering
- Different model types
- Different confidence thresholds
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_images(output_dir, count=100, size=(640, 480)):
    """
    Create test images for benchmarking.

    Args:
        output_dir: Directory to save test images
        count: Number of images to create
        size: Image size (width, height)

    Returns:
        List of created image paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    logger.info(f"Creating {count} test images...")

    for i in range(count):
        # Create random image
        img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Save image
        img_path = output_dir / f"test_image_{i:04d}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))

    logger.info(f"Created {len(image_paths)} test images in {output_dir}")
    return image_paths


def benchmark_filter_worker(image_paths, worker_count):
    """
    Benchmark FilterWorker with different worker counts.

    Args:
        image_paths: List of image paths to filter
        worker_count: Number of worker threads

    Returns:
        Dictionary with benchmark results
    """
    from anylabeling.views.labeling.widgets.image_filter_dialog import FilterWorker

    # Create a mock model manager (simplified for benchmarking)
    class MockModel:
        def predict_shapes(self, image_array, image_path):
            # Simulate some processing time
            time.sleep(0.01)
            # Return mock result
            class MockResult:
                shapes = []
            return MockResult()

    class MockModelManager:
        loaded_model_config = {"model": MockModel()}

    model_manager = MockModelManager()

    logger.info(f"Benchmarking FilterWorker with {worker_count} workers...")
    start_time = time.time()

    # Create worker
    worker = FilterWorker(
        image_paths,
        model_manager,
        min_confidence=0.5,
        max_confidence=1.0,
        selected_classes=None,
        count_mode="any",
        count_value=1,
        max_workers=worker_count,
    )

    # Run filtering
    worker.run()

    elapsed_time = time.time() - start_time

    return {
        "worker_count": worker_count,
        "image_count": len(image_paths),
        "elapsed_time": elapsed_time,
        "images_per_second": len(image_paths) / elapsed_time if elapsed_time > 0 else 0,
    }


def benchmark_cache_performance(image_paths, iterations=3):
    """
    Benchmark filter result cache performance.

    Args:
        image_paths: List of image paths
        iterations: Number of iterations to test cache hits

    Returns:
        Dictionary with cache benchmark results
    """
    from anylabeling.utils.image_cache import FilterResultCache
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = FilterResultCache(cache_dir=tmp_dir)

        # Test parameters
        folder_path = str(Path(image_paths[0]).parent)
        model_name = "test_model"
        min_conf = 0.5
        max_conf = 1.0
        classes = None
        count_mode = "any"
        count_value = 1

        results = {
            "cache_misses": [],
            "cache_hits": [],
        }

        # First iteration - cache miss
        start_time = time.time()
        result = cache.get(
            folder_path, model_name, min_conf, max_conf, classes, count_mode, count_value
        )
        miss_time = time.time() - start_time
        results["cache_misses"].append(miss_time)
        assert result is None, "Expected cache miss"

        # Save to cache
        filtered_images = image_paths[:10]  # Mock filtered results
        cache.put(
            folder_path,
            model_name,
            min_conf,
            max_conf,
            classes,
            count_mode,
            count_value,
            filtered_images,
            len(image_paths),
        )

        # Subsequent iterations - cache hits
        for i in range(iterations):
            start_time = time.time()
            result = cache.get(
                folder_path,
                model_name,
                min_conf,
                max_conf,
                classes,
                count_mode,
                count_value,
            )
            hit_time = time.time() - start_time
            results["cache_hits"].append(hit_time)
            assert result is not None, "Expected cache hit"

        # Calculate statistics
        avg_miss = np.mean(results["cache_misses"])
        avg_hit = np.mean(results["cache_hits"])
        speedup = avg_miss / avg_hit if avg_hit > 0 else 0

        return {
            "avg_cache_miss_time": avg_miss,
            "avg_cache_hit_time": avg_hit,
            "speedup": speedup,
            "cache_stats": cache.get_stats(),
        }


def run_benchmarks(output_file=None):
    """Run all filtering benchmarks."""
    results = {
        "timestamp": time.time(),
        "benchmarks": {},
    }

    # Create test images
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test with different dataset sizes
        dataset_sizes = [10, 50, 100]
        for size in dataset_sizes:
            logger.info(f"\n=== Testing with {size} images ===")
            image_paths = create_test_images(tmp_dir, count=size)

            # Test with different worker counts
            worker_counts = [1, 4, 8]
            for workers in worker_counts:
                key = f"filter_worker_{size}_images_{workers}_workers"
                results["benchmarks"][key] = benchmark_filter_worker(
                    image_paths, workers
                )
                logger.info(
                    f"  {workers} workers: {results['benchmarks'][key]['images_per_second']:.2f} images/sec"
                )

        # Test cache performance
        logger.info("\n=== Testing cache performance ===")
        image_paths = create_test_images(tmp_dir, count=50)
        cache_results = benchmark_cache_performance(image_paths)
        results["benchmarks"]["cache_performance"] = cache_results
        logger.info(
            f"  Cache speedup: {cache_results['speedup']:.2f}x"
        )

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")

    return results


def print_summary(results):
    """Print a summary of benchmark results."""
    print("\n" + "=" * 60)
    print("FILTERING BENCHMARK SUMMARY")
    print("=" * 60)

    # Find best performance for each dataset size
    dataset_sizes = {10, 50, 100}
    for size in dataset_sizes:
        print(f"\nDataset: {size} images")
        best_throughput = 0
        best_workers = 0

        for key, result in results["benchmarks"].items():
            if (
                key.startswith("filter_worker_")
                and f"_{size}_images_" in key
            ):
                throughput = result["images_per_second"]
                workers = result["worker_count"]
                speedup = throughput / result["image_count"] * result["elapsed_time"] if result["elapsed_time"] > 0 else 0
                
                print(
                    f"  {workers} workers: {throughput:.2f} images/sec "
                    f"({result['elapsed_time']:.2f}s total)"
                )

                if throughput > best_throughput:
                    best_throughput = throughput
                    best_workers = workers

        if best_throughput > 0:
            print(f"  → Best: {best_workers} workers")

    # Cache performance
    if "cache_performance" in results["benchmarks"]:
        cache = results["benchmarks"]["cache_performance"]
        print(f"\nCache Performance:")
        print(f"  Miss: {cache['avg_cache_miss_time']*1000:.2f}ms")
        print(f"  Hit: {cache['avg_cache_hit_time']*1000:.2f}ms")
        print(f"  Speedup: {cache['speedup']:.2f}x")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark image filtering performance"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark_filtering_results.json",
        help="Output file for results (default: benchmark_filtering_results.json)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run benchmarks
    results = run_benchmarks(args.output)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
