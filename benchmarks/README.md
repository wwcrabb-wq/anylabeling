# AnyLabeling Benchmarks

This directory contains benchmarking scripts for measuring performance improvements in AnyLabeling.

## Status: Not Yet Implemented

The benchmarking infrastructure is **planned but not yet implemented**. This directory serves as a placeholder for future benchmark scripts.

## Planned Benchmarks

### 1. benchmark_inference.py
Measure model inference performance:
- Single image inference time
- Batch inference time
- Throughput (images/second)
- Compare different batch sizes
- Compare different backends (Ultralytics, cv2.dnn, etc.)

### 2. benchmark_io.py
Measure image I/O performance:
- Sequential vs parallel loading
- Different image formats
- Different image sizes
- Cache hit rates

### 3. benchmark_nms.py
Measure NMS performance:
- Pure Python vs Cython (when implemented)
- Different box counts
- Different IoU thresholds

### 4. benchmark_filtering.py
Measure image filtering performance:
- Sequential vs parallel
- Different thread counts
- Different dataset sizes

### 5. run_benchmarks.py
Master script to run all benchmarks and generate report.

## Usage (When Implemented)

```bash
# Run all benchmarks
python benchmarks/run_benchmarks.py

# Run specific benchmark
python benchmarks/benchmark_inference.py

# Run with custom settings
python benchmarks/benchmark_inference.py --batch-size 8 --iterations 100

# Generate report
python benchmarks/run_benchmarks.py --output report.html
```

## Requirements

When implementing benchmarks:
- Use consistent test datasets
- Run multiple iterations for statistical significance
- Measure both mean and percentile latencies (p50, p95, p99)
- Include system information in reports
- Compare against baseline measurements

## Example Output

```
=== Inference Benchmark ===
Model: YOLOv8n
Backend: Ultralytics
Device: CUDA

Single Image:
  Mean: 15.2ms
  P95: 18.1ms
  P99: 22.3ms

Batch Size 4:
  Mean: 42.1ms (10.5ms/image)
  P95: 45.8ms
  Speedup: 1.45x

Batch Size 8:
  Mean: 76.4ms (9.6ms/image)
  P95: 82.1ms
  Speedup: 1.58x
```

## Contributing

When implementing benchmarks:
1. Use `time.perf_counter()` for timing
2. Warm up models before benchmarking
3. Run multiple iterations (at least 10)
4. Report both mean and variance
5. Include system specs in output
6. Make benchmarks reproducible
7. Document any special setup required
