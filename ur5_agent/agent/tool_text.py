# agent/tool_text.py — recover Ollama tool calls when the model writes text instead

import json
import re

from agent.schemas import agent_tool_schemas

_AGENT_TOOL_NAMES = {t["name"] for t in agent_tool_schemas()}

_MOTION_TOOLS = {
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "move_forward",
    "move_backward",
}

_JSON_TOOL = re.compile(
    r'\{\s*"name"\s*:\s*"([a-z_][a-z0-9_]*)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
    re.IGNORECASE | re.DOTALL,
)

_PAREN_TOOL = re.compile(
    r"[-]?\s*\b([a-z_][a-z0-9_]*)\s*\(\s*([^)]*)\s*\)",
    re.IGNORECASE,
)


def _normalize_args(tool_name: str, raw_args: str) -> dict:
    raw = (raw_args or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    if tool_name in _MOTION_TOOLS:
        try:
            return {"distance_m": float(raw.strip().strip(",").strip('"').strip("'"))}
        except ValueError:
            return {}
    return {}


def recover_tool_calls_from_text(text: str | None) -> list[dict]:
    """
    Parse pseudo tool calls from assistant text, e.g. move_left(0.05) or
    {"name": "move_right", "arguments": {"distance_m": 0.05}}.
    """
    if not text:
        return []

    found: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(name: str, args: dict) -> None:
        name = name.strip().lower()
        if name not in _AGENT_TOOL_NAMES:
            return
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            return
        seen.add(key)
        found.append({"name": name, "arguments": args})

    for match in _JSON_TOOL.finditer(text):
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        add(match.group(1), args)

    if found:
        return found

    for match in _PAREN_TOOL.finditer(text):
        name = match.group(1)
        if name not in _AGENT_TOOL_NAMES:
            continue
        add(name, _normalize_args(name, match.group(2)))

    return found
