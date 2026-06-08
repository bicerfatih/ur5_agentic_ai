# agent/schemas.py — tool definitions for Claude and Ollama backends

from config.settings import AGENT_ALLOW_MOVE_HOME
from robot.tools import TOOL_SCHEMAS

# Agent must not use these for error recovery (use Tool Console / pendant instead).
_AGENT_BLOCKED_TOOLS = frozenset(
    {
        "move_home",
        "move_joint",
        "release_rtde_control",
        "run_urp_program",
    }
)


def agent_tool_schemas() -> list[dict]:
    """Tool list exposed to Agentic AI (no home / joint / urp / rtde release)."""
    if AGENT_ALLOW_MOVE_HOME:
        blocked = _AGENT_BLOCKED_TOOLS - {"move_home", "move_joint"}
    else:
        blocked = _AGENT_BLOCKED_TOOLS
    return [t for t in TOOL_SCHEMAS if t["name"] not in blocked]


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
        for t in agent_tool_schemas()
    ]
