"""Record provider-reported token usage for one mapping scene."""
from __future__ import annotations

import json
from pathlib import Path


def count(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def normalized_usage(backend: str, raw: dict | None) -> dict | None:
    if not isinstance(raw, dict):
        return None
    if backend == "anthropic":
        uncached = count(raw.get("input_tokens"))
        cache_written = count(raw.get("cache_creation_input_tokens")) or 0
        cache_read = count(raw.get("cache_read_input_tokens")) or 0
        input_tokens = uncached + cache_written + cache_read if uncached is not None else None
        output_tokens = count(raw.get("output_tokens"))
        details = raw.get("output_tokens_details") or {}
        reasoning_tokens = count(details.get("thinking_tokens")) if isinstance(details, dict) else None
        total_tokens = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
        return {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": total_tokens, "cached_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_written,
            "reasoning_output_tokens": reasoning_tokens,
        }
    if backend == "gemini":
        input_tokens = count(raw.get("promptTokenCount"))
        visible_output = count(raw.get("candidatesTokenCount"))
        reasoning_tokens = count(raw.get("thoughtsTokenCount")) or 0
        output_tokens = visible_output + reasoning_tokens if visible_output is not None else None
        total_tokens = count(raw.get("totalTokenCount"))
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": count(raw.get("cachedContentTokenCount")) or 0,
            "reasoning_output_tokens": reasoning_tokens,
            "visible_output_tokens": visible_output,
            "tool_use_prompt_tokens": count(raw.get("toolUsePromptTokenCount")) or 0,
        }
    input_tokens = count(raw.get("input_tokens"))
    output_tokens = count(raw.get("output_tokens"))
    input_details = raw.get("input_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    if backend == "openai_chat":
        input_tokens = count(raw.get("prompt_tokens"))
        output_tokens = count(raw.get("completion_tokens"))
        input_details = raw.get("prompt_tokens_details") or {}
        output_details = raw.get("completion_tokens_details") or {}
    cached_input_tokens = count(input_details.get("cached_tokens")) or 0 if isinstance(input_details, dict) else 0
    reasoning_output_tokens = count(output_details.get("reasoning_tokens")) if isinstance(output_details, dict) else None
    total_tokens = count(raw.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
    }


class TokenUsageRecorder:
    def __init__(self, output_dir: Path, backend: str):
        self.path = output_dir / "token_usage.json"
        self.backend = backend
        self.turns = []

    def record(self, turn: int, raw: dict | None) -> dict | None:
        usage = normalized_usage(self.backend, raw)
        self.turns.append({"turn": turn, "usage": usage, "provider_usage": raw if isinstance(raw, dict) else None})
        self.save()
        return usage

    def summary(self) -> dict:
        known = [item["usage"] for item in self.turns if item["usage"] is not None]
        fields = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "cache_creation_input_tokens", "reasoning_output_tokens", "visible_output_tokens", "tool_use_prompt_tokens")
        totals = {field: sum(item.get(field) or 0 for item in known) for field in fields if any(item.get(field) is not None for item in known)}
        return {
            "backend": self.backend,
            "request_count": len(self.turns),
            "requests_with_usage": len(known),
            "requests_without_usage": len(self.turns) - len(known),
            "usage_complete": len(known) == len(self.turns) and all(item.get("total_tokens") is not None for item in known),
            **totals,
        }

    def save(self) -> None:
        data = {**self.summary(), "turns": self.turns}
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        temporary.replace(self.path)
