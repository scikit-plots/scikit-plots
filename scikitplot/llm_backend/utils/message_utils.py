# llm_backend/utils/message_utils.py
from typing import Dict, List

from ..base import LLMBackend

# ------------------------
# Utility: Truncate Messages by Token Limit
# ------------------------


def truncate_messages(
    messages: List[Dict], backend: LLMBackend, max_tokens: int = 2048
) -> List[Dict]:
    """
    Truncate a list of chat messages to fit within a specified token limit.

    This function uses the backend's token counting method to ensure the
    total token count of messages does not exceed `max_tokens`.
    If over the limit, it removes messages from the start (oldest user+assistant pairs)
    until the limit is satisfied or only the system message and last user remain.

    Args:
        messages (List[Dict]): List of messages, each a dict with 'role' and 'content'.
        backend (LLMBackend): LLM backend instance implementing `count_tokens()`.
        max_tokens (int): Maximum allowed tokens (default: 2048).

    Returns
    -------
        List[Dict]: Truncated list of messages within token constraints.
    """
    while backend.count_tokens(messages) > max_tokens and len(messages) > 2:
        # Remove the oldest user and assistant messages (first two messages after system)
        messages = messages[2:]
    return messages
