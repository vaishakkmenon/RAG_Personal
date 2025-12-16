"""
Prompt Injection Guard using Llama Prompt Guard 2.

This service checks user input for prompt injection/jailbreak attempts
before sending to the main LLM. Uses Groq's hosted Llama Prompt Guard 2.

Model options:
- llama-prompt-guard-2-86m: 99.8% AUC, 97.5% recall (recommended)
- llama-prompt-guard-2-22m: 99.5% AUC, 88.7% recall (faster)
"""

import logging
import re
import time
import hashlib
from typing import Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq SDK not available - prompt guard disabled")

# Import metrics with graceful degradation
try:
    from ..metrics import (
        prompt_guard_checks_total,
        prompt_guard_blocked_total,
        prompt_guard_api_latency_seconds,
        prompt_guard_cache_operations_total,
        prompt_guard_errors_total,
        prompt_guard_retries_total,
        prompt_guard_context_size_chars,
    )
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.warning("Metrics not available - prompt guard metrics disabled")


class PromptGuard:
    """Guard against prompt injection attacks using Llama Prompt Guard 2."""

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-prompt-guard-2-86m",
        enabled: bool = True,
        fail_open: bool = True,
        timeout_seconds: float = 3.0,
        max_retries: int = 2,
        cache_ttl_seconds: int = 3600,
        cache_max_size: int = 1000,
    ):
        """
        Initialize prompt guard.

        Args:
            api_key: Groq API key
            model: Prompt guard model to use
            enabled: Whether guard is enabled
            fail_open: If True, allow requests on error; if False, block on error
            timeout_seconds: Timeout for API calls
            max_retries: Maximum retry attempts
            cache_ttl_seconds: Cache TTL in seconds
            cache_max_size: Maximum cache entries
        """
        self.model = model
        self.enabled = enabled and GROQ_AVAILABLE
        self.fail_open = fail_open
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.cache_ttl_seconds = cache_ttl_seconds
        self._client = None

        # Simple in-memory cache: {hash: (result, timestamp)}
        self._cache: Dict[str, tuple[Dict[str, Any], float]] = {}
        self._cache_max_size = cache_max_size

        if self.enabled and api_key:
            try:
                self._client = Groq(api_key=api_key, timeout=timeout_seconds)
                logger.info(f"PromptGuard initialized with model: {model}, timeout: {timeout_seconds}s")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.enabled = False
        elif self.enabled:
            logger.warning("PromptGuard disabled: No API key provided")
            self.enabled = False

    def _get_cache_key(self, user_input: str) -> str:
        """Generate cache key from user input."""
        return hashlib.sha256(user_input.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache if valid."""
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl_seconds:
                logger.debug(f"PromptGuard cache hit for key: {cache_key[:8]}...")
                return result
            else:
                # Expired, remove it
                del self._cache[cache_key]
        return None

    def _put_in_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Store result in cache with LRU eviction."""
        # Simple LRU: if full, remove oldest entry
        if len(self._cache) >= self._cache_max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        self._cache[cache_key] = (result, time.time())
        logger.debug(f"PromptGuard cached result for key: {cache_key[:8]}...")

    def check_input(
        self,
        user_input: str,
        conversation_history: Optional[list] = None,
        max_history_turns: int = 3
    ) -> Dict[str, Any]:
        """
        Check if input contains prompt injection, with optional conversation context.

        Multi-turn injection attacks can split malicious content across multiple messages.
        This method can check recent conversation history to detect such attacks.

        Args:
            user_input: The user's current query to check
            conversation_history: Optional list of recent conversation turns
                                 Format: [{"role": "user"|"assistant", "content": "..."}]
            max_history_turns: Maximum number of recent turns to include (default: 3)

        Returns:
            Dict with keys:
            - safe: bool - Whether input is safe
            - label: str - Classification result
            - blocked: bool - Whether request should be blocked
        """
        if not self.enabled or not self._client:
            return {"safe": True, "label": "GUARD_DISABLED", "blocked": False}

        # Build input with conversation context for multi-turn attack detection
        input_to_check = user_input
        if conversation_history and len(conversation_history) > 0:
            # Take last N turns (limit to max_history_turns)
            recent_history = conversation_history[-max_history_turns:]

            # Build context string: concatenate recent messages
            context_parts = []
            for turn in recent_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                context_parts.append(f"{role}: {content}")

            # Append current input
            context_parts.append(f"user: {user_input}")

            # Join with newlines for clarity
            input_to_check = "\n".join(context_parts)

            logger.debug(
                f"PromptGuard checking with {len(recent_history)} turns of context "
                f"({len(input_to_check)} chars total)"
            )

        # Track context size
        if METRICS_ENABLED:
            prompt_guard_context_size_chars.observe(len(input_to_check))

        # Check cache first (using the full context as cache key)
        cache_key = self._get_cache_key(input_to_check)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            if METRICS_ENABLED:
                prompt_guard_cache_operations_total.labels(result="hit").inc()
                prompt_guard_checks_total.labels(
                    result="blocked" if cached_result["blocked"] else "safe"
                ).inc()
            return cached_result

        # Cache miss
        if METRICS_ENABLED:
            prompt_guard_cache_operations_total.labels(result="miss").inc()

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Track retry attempts
                    if METRICS_ENABLED:
                        prompt_guard_retries_total.labels(attempt=str(attempt)).inc()

                    # Exponential backoff: 0.1s, 0.2s, 0.4s...
                    backoff = 0.1 * (2 ** (attempt - 1))
                    logger.debug(f"PromptGuard retry {attempt}/{self.max_retries} after {backoff}s")
                    time.sleep(backoff)

                # Track API call latency
                api_start = time.time()
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": input_to_check}],
                )
                api_duration = time.time() - api_start

                if METRICS_ENABLED:
                    prompt_guard_api_latency_seconds.observe(api_duration)

                # Parse response - model returns simple binary classification
                result = response.choices[0].message.content.strip()
                result_lower = result.lower()

                logger.debug(f"PromptGuard raw response: '{result}'")

                # Llama Prompt Guard 2 returns one of:
                # - "benign" or "malicious" (text labels)
                # - "LABEL_0" (benign) or "LABEL_1" (malicious) (numeric labels)
                is_malicious = ("malicious" in result_lower or
                               result.strip() == "LABEL_1")
                is_benign = ("benign" in result_lower or
                            result.strip() == "LABEL_0")

                # If we got an unexpected response format, apply fail-open/fail-closed policy
                if not is_malicious and not is_benign:
                    logger.warning(
                        f"PromptGuard: Unexpected response format '{result}', "
                        f"defaulting to fail_open={self.fail_open}"
                    )
                    is_malicious = not self.fail_open
                    is_benign = self.fail_open

                if is_malicious:
                    logger.info(f"PromptGuard: BLOCKED - '{result}'")

                guard_result = {
                    "safe": is_benign and not is_malicious,
                    "label": result,
                    "blocked": is_malicious
                }

                # Track result metrics
                if METRICS_ENABLED:
                    prompt_guard_checks_total.labels(
                        result="blocked" if is_malicious else "safe"
                    ).inc()

                    if is_malicious:
                        prompt_guard_blocked_total.labels(label=result).inc()

                # Cache the result
                self._put_in_cache(cache_key, guard_result)

                return guard_result

            except Exception as e:
                last_exception = e
                logger.warning(f"PromptGuard attempt {attempt + 1}/{self.max_retries + 1} failed: {e}")

                # Track error type
                if METRICS_ENABLED:
                    error_type = "timeout" if "timeout" in str(e).lower() else "api_error"
                    prompt_guard_errors_total.labels(error_type=error_type).inc()

                # If this was the last attempt, fall through to error handling
                if attempt >= self.max_retries:
                    break

        # All retries exhausted
        logger.error(f"PromptGuard check failed after {self.max_retries + 1} attempts: {last_exception}")

        # Track final error
        if METRICS_ENABLED:
            prompt_guard_checks_total.labels(result="safe" if self.fail_open else "blocked").inc()
            if not self.fail_open:
                prompt_guard_blocked_total.labels(label="ERROR").inc()

        # Fail-open or fail-closed based on configuration
        if self.fail_open:
            return {"safe": True, "label": "ERROR", "blocked": False}
        else:
            return {"safe": False, "label": "ERROR", "blocked": True}


# Singleton instance
_prompt_guard_instance: Optional[PromptGuard] = None


def get_prompt_guard() -> PromptGuard:
    """
    Get singleton PromptGuard instance.

    Returns:
        PromptGuard instance configured from settings
    """
    global _prompt_guard_instance

    if _prompt_guard_instance is None:
        from ..settings import settings

        _prompt_guard_instance = PromptGuard(
            api_key=settings.llm.groq_api_key,
            model=settings.prompt_guard.model,
            enabled=settings.prompt_guard.enabled,
            fail_open=settings.prompt_guard.fail_open,
            timeout_seconds=settings.prompt_guard.timeout_seconds,
            max_retries=settings.prompt_guard.max_retries,
            cache_ttl_seconds=settings.prompt_guard.cache_ttl_seconds,
            cache_max_size=settings.prompt_guard.cache_max_size,
        )

    return _prompt_guard_instance


def reset_prompt_guard() -> None:
    """Reset singleton (useful for testing)."""
    global _prompt_guard_instance
    _prompt_guard_instance = None


__all__ = ["PromptGuard", "get_prompt_guard", "reset_prompt_guard"]
