"""Helpers for recognizing structured COOP2 repair messages."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from coop2_repair.message_protocol import (
    COOP2_REPAIR_CONTENT_TYPE,
    COOP2_REPAIR_MESSAGE_TYPE,
    MESSAGE_TYPE_METADATA_KEY,
)


def is_coop2_repair_content(content: Any) -> bool:
    """Return True when a message payload is a structured COOP2 repair request."""
    return isinstance(content, dict) and content.get("type") == COOP2_REPAIR_CONTENT_TYPE


def is_coop2_repair_message(message: Any) -> bool:
    """Return True when a message record carries a COOP2 repair request."""
    if not isinstance(message, dict):
        return False
    metadata = message.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if metadata.get(MESSAGE_TYPE_METADATA_KEY) == COOP2_REPAIR_MESSAGE_TYPE:
        return True
    return is_coop2_repair_content(message.get("content"))


def has_coop2_repair_message(messages: Optional[Iterable[Dict[str, Any]]]) -> bool:
    """Return True when any message in an iterable is a COOP2 repair request."""
    return any(is_coop2_repair_message(message) for message in messages or [])
