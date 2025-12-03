"""
Helper functions for session storage.
"""

from typing import List, Dict

def estimate_tokens(text: str) -> int:
    """Estimate token count using word-based heuristic.
    
    More accurate than character count:
    - English: ~0.75 tokens per word
    - Accounts for punctuation and whitespace
    
    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Split on whitespace and count words
    words = len(text.split())
    return int(words * 0.75) if words > 0 else len(text) // 4


def truncate_history(
    history: List[Dict[str, str]],
    max_tokens: int = 250,
    max_turns: int = 5
) -> List[Dict[str, str]]:
    """Truncate conversation history to fit token budget.

    Keeps most recent turns that fit within budget. Ensures complete
    user-assistant pairs (no orphaned messages).

    Args:
        history: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens to allocate
        max_turns: Maximum number of messages to keep

    Returns:
        Truncated history
    """
    if not history:
        return []

    # Truncate by turn count first
    if len(history) > max_turns:
        history = history[-max_turns:]

    # Truncate by token count
    truncated = []
    tokens_used = 0

    # Iterate from most recent to oldest
    for i in range(len(history) - 1, -1, -1):
        msg = history[i]
        msg_tokens = estimate_tokens(msg.get("content", ""))

        if tokens_used + msg_tokens > max_tokens and truncated:
            # Stop adding more messages
            break

        truncated.insert(0, msg)
        tokens_used += msg_tokens

    # Remove orphaned assistant message at start
    if truncated and truncated[0].get("role") == "assistant":
        truncated = truncated[1:]

    return truncated


def mask_session_id(session_id: str) -> str:
    """Mask session ID for secure logging.

    Shows only first 8 characters to prevent session hijacking via logs.

    Args:
        session_id: Full session ID

    Returns:
        Masked ID (e.g., "abc12345***")
    """
    if not session_id or len(session_id) < 8:
        return "***"
    return f"{session_id[:8]}***"
