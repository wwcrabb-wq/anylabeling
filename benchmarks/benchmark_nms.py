"""
Benchmark NMS implementations.

Usage:
    python benchmarks/benchmark_nms.py --output results.json
"""

import argparse
import time
import numpy as np
import json
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_boxes(num_boxes: int) -> tuple:
    """Generate random test boxes and scores."""
    boxes = np.random.rand(num_boxes, 4).astype(np.float32) * 100
    # Ensure x2 > x1 and y2 > y1
    boxes[:, 2] = boxes[:, 0] + np.random.rand(num_boxes) * 50
    boxes[:, 3] = boxes[:, 1] + np.random.rand(num_boxes) * 50
    scores = np.random.rand(num_boxes).astype(np.float32)
    return boxes, scores


def benchmark_python_nms(boxes, scores, iou_threshold=0.5, iterations=100) -> Dict:
    """Benchmark pure Python NMS."""
    from anylabeling.extensions.fallbacks import fast_nms_python
    
    logger.info("Benchmarking Python NMS...")
    timings = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = fast_nms_python(boxes, scores, iou_threshold)
        timings.append((time.perf_counter() - start) * 1000)
    
    return {
        "implementation": "python",
        "mean_ms": np.mean(timings),
        "std_ms": np.std(timings),
        "min_ms": np.min(timings),
        "max_ms": np.max(timings),
    }


def benchmark_cython_nms(boxes, scores, iou_threshold=0.5, iterations=100) -> Dict:
    """Benchmark Cython NMS."""
    try:
        from anylabeling.extensions import fast_nms, CYTHON_NMS_AVAILABLE
        
        if not CYTHON_NMS_AVAILABLE:
            return {"implementation": "cython", "status": "unavailable"}
        
        logger.info("Benchmarking Cython NMS...")
        timings = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            _ = fast_nms(boxes, scores, iou_threshold)
            timings.append((time.perf_counter() - start) * 1000)
        
        return {
            "implementation": "cython",
            "mean_ms": np.mean(timings),
            "std_ms": np.std(timings),
            "min_ms": np.min(timings),
            "max_ms": np.max(timings),
        }
    except Exception as e:
        logger.warning(f"Cython NMS failed: {e}")
        return {"implementation": "cython", "status": "error", "error": str(e)}


def benchmark_opencv_nms(boxes, scores, iou_threshold=0.5, iterations=100) -> Dict:
    """Benchmark OpenCV NMS."""
    try:
        import cv2
        
        logger.info("Benchmarking OpenCV NMS...")
        timings = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            # OpenCV NMS requires boxes in [x, y, w, h] format
            boxes_xywh = boxes.copy()
            boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]
            boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]
            _ = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), 0.0, iou_threshold)
            timings.append((time.perf_counter() - start) * 1000)
        
        return {
            "implementation": "opencv",
            "mean_ms": np.mean(timings),
            "std_ms": np.std(timings),
            "min_ms": np.min(timings),
            "max_ms": np.max(timings),
        }
    except Exception as e:
        logger.warning(f"OpenCV NMS failed: {e}")
        return {"implementation": "opencv", "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Benchmark NMS implementations")
    parser.add_argument("--output", type=str, default="nms_benchmark_results.json")
    parser.add_argument("--box-counts", type=int, nargs="+", default=[100, 500, 1000, 5000])
    
    args = parser.parse_args()
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": []
    }
    
    for num_boxes in args.box_counts:
        logger.info(f"\nBenchmarking with {num_boxes} boxes...")
        boxes, scores = generate_test_boxes(num_boxes)
        
        benchmark = {
            "num_boxes": num_boxes,
            "results": [
                benchmark_python_nms(boxes, scores),
                benchmark_cython_nms(boxes, scores),
                benchmark_opencv_nms(boxes, scores),
            ]
        }
        results["benchmarks"].append(benchmark)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n=== NMS Benchmark Summary ===")
    for benchmark in results["benchmarks"]:
        print(f"\nBox count: {benchmark['num_boxes']}")
        for result in benchmark["results"]:
            if "mean_ms" in result:
                print(f"  {result['implementation']}: {result['mean_ms']:.2f} ms")


if __name__ == "__main__":
    main()
