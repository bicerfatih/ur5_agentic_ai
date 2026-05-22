# agent/schemas.py — tool definitions for Claude and Ollama backends

from robot.tools import TOOL_SCHEMAS


def ollama_tools() -> list[dict]:
    """Convert Anthropic-style tool schemas to Ollama function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOL_SCHEMAS
    ]
