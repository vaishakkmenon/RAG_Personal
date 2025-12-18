"""
Rate limiter for LLM API calls.

Implements token bucket algorithm to limit request rates.
"""

import time
import threading
from collections import deque
from typing import Optional


class RateLimiter:
    """Simple rate limiter using sliding window approach"""

    def __init__(
        self,
        requests_per_minute: int = 30,
        requests_per_day: int = 14400,
    ):
        """Initialize rate limiter

        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_day: Maximum requests per day
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day

        # Track requests in sliding windows
        self.minute_requests = deque(maxlen=requests_per_minute)
        self.day_requests = deque(maxlen=requests_per_day)

        # Thread safety
        self.lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission to make a request

        Blocks until rate limit allows the request or timeout is reached.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if permission granted, False if timeout
        """
        start_time = time.time()

        while True:
            with self.lock:
                now = time.time()

                # Clean old requests
                self._clean_old_requests(now)

                # Check if we can make a request
                minute_ok = len(self.minute_requests) < self.requests_per_minute
                day_ok = len(self.day_requests) < self.requests_per_day

                if minute_ok and day_ok:
                    # Grant permission
                    self.minute_requests.append(now)
                    self.day_requests.append(now)
                    return True

                # Calculate wait time
                if not minute_ok and self.minute_requests:
                    # Wait until oldest minute request expires
                    oldest = self.minute_requests[0]
                    wait_time = 60 - (now - oldest)
                elif not day_ok and self.day_requests:
                    # Wait until oldest day request expires
                    oldest = self.day_requests[0]
                    wait_time = 86400 - (now - oldest)
                else:
                    wait_time = 1  # Shouldn't happen, but just in case

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

                # Don't wait longer than remaining timeout
                wait_time = min(wait_time, timeout - elapsed)

            # Sleep before retrying
            time.sleep(min(wait_time, 1))

    def _clean_old_requests(self, now: float):
        """Remove requests older than tracking windows"""
        # Clean minute window (keep last 60 seconds)
        while self.minute_requests and now - self.minute_requests[0] > 60:
            self.minute_requests.popleft()

        # Clean day window (keep last 24 hours)
        while self.day_requests and now - self.day_requests[0] > 86400:
            self.day_requests.popleft()

    def get_stats(self) -> dict:
        """Get current rate limit statistics"""
        with self.lock:
            now = time.time()
            self._clean_old_requests(now)

            return {
                "requests_last_minute": len(self.minute_requests),
                "requests_per_minute_limit": self.requests_per_minute,
                "requests_last_day": len(self.day_requests),
                "requests_per_day_limit": self.requests_per_day,
                "minute_utilization": len(self.minute_requests)
                / self.requests_per_minute,
                "day_utilization": len(self.day_requests) / self.requests_per_day,
            }


class NoOpRateLimiter:
    """No-op rate limiter that always allows requests (for Ollama)"""

    def acquire(self, timeout: Optional[float] = None) -> bool:
        return True

    def get_stats(self) -> dict:
        return {
            "requests_last_minute": 0,
            "requests_per_minute_limit": float("inf"),
            "requests_last_day": 0,
            "requests_per_day_limit": float("inf"),
            "minute_utilization": 0.0,
            "day_utilization": 0.0,
        }
