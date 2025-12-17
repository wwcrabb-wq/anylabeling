"""
Master script to run all benchmarks and generate comprehensive report.

Usage:
    python benchmarks/run_benchmarks.py --output-dir benchmark_results/
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import time
import platform
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_system_info() -> dict:
    """Collect system information."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    
    # Try to get GPU info
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info["gpu"] = result.stdout.strip()
    except:
        info["gpu"] = "Not available"
    
    return info


def run_benchmark_script(script_name: str, args: list, output_dir: Path) -> dict:
    """Run a benchmark script and return results."""
    logger.info(f"Running {script_name}...")
    
    output_file = output_dir / f"{Path(script_name).stem}_results.json"
    cmd = [sys.executable, f"benchmarks/{script_name}", "--output", str(output_file)] + args
    
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - start
        
        if result.returncode != 0:
            logger.error(f"Benchmark failed: {result.stderr}")
            return {"status": "failed", "error": result.stderr}
        
        # Load results
        if output_file.exists():
            with open(output_file) as f:
                data = json.load(f)
            return {"status": "success", "elapsed": elapsed, "data": data}
        else:
            return {"status": "failed", "error": "No output file generated"}
    
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def generate_html_report(results: dict, output_path: Path):
    """Generate HTML report from all benchmark results."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>AnyLabeling Performance Benchmarks</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .info { background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>AnyLabeling Performance Benchmarks</h1>
    <div class="timestamp">Generated: {timestamp}</div>
    
    <div class="info">
        <h3>System Information</h3>
        <ul>
            <li><strong>Platform:</strong> {platform}</li>
            <li><strong>Python:</strong> {python_version}</li>
            <li><strong>Processor:</strong> {processor}</li>
            <li><strong>GPU:</strong> {gpu}</li>
        </ul>
    </div>
    
    <h2>Benchmark Results</h2>
    {benchmark_sections}
</body>
</html>
    """.format(
        timestamp=results["timestamp"],
        platform=results["system_info"]["platform"],
        python_version=results["system_info"]["python_version"],
        processor=results["system_info"]["processor"],
        gpu=results["system_info"].get("gpu", "N/A"),
        benchmark_sections="<p>See individual JSON files for detailed results.</p>"
    )
    
    with open(output_path, "w") as f:
        f.write(html)
    
    logger.info(f"HTML report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all performance benchmarks")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--skip", nargs="+", default=[],
                       help="Benchmarks to skip (e.g., inference io)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Starting benchmark suite...")
    logger.info(f"Output directory: {output_dir}")
    
    # Collect system info
    system_info = get_system_info()
    
    # Results container
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": system_info,
        "benchmarks": {}
    }
    
    # Define benchmarks to run
    benchmarks = [
        ("benchmark_nms.py", []),
        ("benchmark_io.py", ["--images", "sample_images", "--limit", "50"]),
    ]
    
    # Run each benchmark
    for script, script_args in benchmarks:
        script_name = Path(script).stem
        if script_name in args.skip:
            logger.info(f"Skipping {script_name}")
            continue
        
        result = run_benchmark_script(script, script_args, output_dir)
        results["benchmarks"][script_name] = result
    
    # Save combined results
    combined_results_path = output_dir / "combined_results.json"
    with open(combined_results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Combined results saved to {combined_results_path}")
    
    # Generate HTML report
    html_report_path = output_dir / "benchmark_report.html"
    generate_html_report(results, html_report_path)
    
    logger.info("\nBenchmark suite complete!")
    logger.info(f"Results directory: {output_dir.absolute()}")
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    for name, result in results["benchmarks"].items():
        status = result.get("status", "unknown")
        elapsed = result.get("elapsed", 0)
        print(f"{name}: {status} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
