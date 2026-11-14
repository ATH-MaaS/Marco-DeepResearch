"""Non-invasive token usage tracker for LLMClient.

Monkey-patches the underlying AsyncOpenAI client to capture token usage
from API responses without modifying llm/client.py.

Uses contextvars for concurrency-safe per-task tracking: each asyncio task
gets its own UsageTracker via set_current_tracker(), and the patched create
method routes usage records to whichever tracker is active in the current task.

Usage:
    from evaluation.usage_tracker import UsageTracker, attach_usage_tracking, set_current_tracker

    # Patch once at setup (idempotent)
    attach_usage_tracking(llm_client)

    # In each concurrent task:
    tracker = UsageTracker()
    set_current_tracker(tracker)
    # ... run agent / llm calls ...
    usage = tracker.snapshot()
"""

from __future__ import annotations

import contextvars
from typing import Optional

# Per-task tracker (copied on asyncio.create_task, isolated per coroutine chain)
_current_tracker: contextvars.ContextVar[Optional["UsageTracker"]] = contextvars.ContextVar(
    "_current_tracker", default=None
)


class UsageTracker:
    """Accumulates token usage across multiple LLM API calls."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.num_calls = 0

    def record(self, usage):
        """Record usage from an OpenAI ChatCompletion response.usage object."""
        if usage is None:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.total_tokens += getattr(usage, "total_tokens", 0) or 0
        self.num_calls += 1

    def snapshot(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "num_calls": self.num_calls,
        }


def set_current_tracker(tracker: Optional[UsageTracker]):
    """Set the active UsageTracker for the current asyncio task context."""
    _current_tracker.set(tracker)


def attach_usage_tracking(llm_client):
    """Monkey-patch an LLMClient to route token usage to the current task's tracker.

    Call once per LLMClient instance. Idempotent (safe to call multiple times).
    The patched method looks up the active tracker via contextvars, so different
    concurrent tasks can each have their own tracker without interference.
    """
    api_client = llm_client.client  # AsyncOpenAI instance
    original_create = api_client.chat.completions.create

    # Avoid double-patching
    if getattr(original_create, "_usage_tracked", False):
        return

    async def tracked_create(*args, **kwargs):
        response = await original_create(*args, **kwargs)
        tracker = _current_tracker.get(None)
        if tracker is not None and hasattr(response, "usage") and response.usage:
            tracker.record(response.usage)
        return response

    tracked_create._usage_tracked = True
    api_client.chat.completions.create = tracked_create
