"""Performance monitoring and optimization utilities."""

import logging
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any
import functools

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and log performance metrics."""

    def __init__(self):
        self._timings = {}
        self._counts = {}

    @contextmanager
    def measure(self, operation: str):
        """
        Context manager to measure operation time.

        Args:
            operation: Name of the operation being measured

        Example:
            with perf_monitor.measure("image_loading"):
                load_image(path)
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self._record_timing(operation, elapsed)

    def _record_timing(self, operation: str, elapsed: float):
        """Record timing for an operation."""
        if operation not in self._timings:
            self._timings[operation] = []
            self._counts[operation] = 0

        self._timings[operation].append(elapsed)
        self._counts[operation] += 1

        # Keep only last 100 measurements to avoid memory growth
        if len(self._timings[operation]) > 100:
            self._timings[operation] = self._timings[operation][-100:]

    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics.

        Args:
            operation: Specific operation to get stats for, or None for all

        Returns:
            Dictionary with performance statistics
        """
        if operation:
            if operation not in self._timings:
                return {}
            timings = self._timings[operation]
            return {
                "count": self._counts[operation],
                "total_time": sum(timings),
                "avg_time": sum(timings) / len(timings) if timings else 0,
                "min_time": min(timings) if timings else 0,
                "max_time": max(timings) if timings else 0,
            }
        else:
            return {op: self.get_stats(op) for op in self._timings.keys()}

    def log_stats(self, operation: Optional[str] = None):
        """
        Log performance statistics.

        Args:
            operation: Specific operation to log, or None for all
        """
        stats = self.get_stats(operation)
        if operation:
            if stats:
                logger.info(
                    f"Performance [{operation}]: "
                    f"count={stats['count']}, "
                    f"avg={stats['avg_time'] * 1000:.2f}ms, "
                    f"min={stats['min_time'] * 1000:.2f}ms, "
                    f"max={stats['max_time'] * 1000:.2f}ms"
                )
        else:
            for op, op_stats in stats.items():
                logger.info(
                    f"Performance [{op}]: "
                    f"count={op_stats['count']}, "
                    f"avg={op_stats['avg_time'] * 1000:.2f}ms, "
                    f"min={op_stats['min_time'] * 1000:.2f}ms, "
                    f"max={op_stats['max_time'] * 1000:.2f}ms"
                )

    def reset(self, operation: Optional[str] = None):
        """
        Reset performance statistics.

        Args:
            operation: Specific operation to reset, or None for all
        """
        if operation:
            if operation in self._timings:
                del self._timings[operation]
                del self._counts[operation]
        else:
            self._timings.clear()
            self._counts.clear()


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_monitor


def timed(operation: Optional[str] = None):
    """
    Decorator to measure function execution time.

    Args:
        operation: Name for the operation (defaults to function name)

    Example:
        @timed("my_function")
        def my_function():
            pass
    """

    def decorator(func):
        op_name = operation or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _global_monitor.measure(op_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
