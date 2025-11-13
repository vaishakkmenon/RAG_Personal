"""
Performance Monitoring - Execution timing and metrics

Provides decorators and utilities for tracking performance metrics.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


def time_execution(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure and log function execution time.

    Works with both sync and async functions.

    Args:
        func: Function to measure

    Returns:
        Wrapped function with timing
    """
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.debug(f"{func.__name__} took {duration_ms:.2f}ms")

        return cast(Callable[..., T], async_wrapper)
    else:

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.debug(f"{func.__name__} took {duration_ms:.2f}ms")

        return cast(Callable[..., T], sync_wrapper)


def time_execution_info(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure and log function execution time at INFO level.

    Useful for high-level operations where timing is important to track.

    Args:
        func: Function to measure

    Returns:
        Wrapped function with timing
    """
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(f"{func.__name__} completed in {duration_ms:.2f}ms")

        return cast(Callable[..., T], async_wrapper)
    else:

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(f"{func.__name__} completed in {duration_ms:.2f}ms")

        return cast(Callable[..., T], sync_wrapper)


class PerformanceTracker:
    """Track performance metrics for operations."""

    def __init__(self, name: str):
        """Initialize tracker.

        Args:
            name: Name of the operation being tracked
        """
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> "PerformanceTracker":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and log."""
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        if exc_type is None:
            logger.debug(f"{self.name} completed in {duration_ms:.2f}ms")
        else:
            logger.warning(
                f"{self.name} failed after {duration_ms:.2f}ms: {exc_type.__name__}"
            )

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time == 0:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
