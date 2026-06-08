# agent/ollama_agent.py — local Ollama agentic loop with tool calling

import json
import os
import time

from ollama import Client

from agent.base_agent import BaseRobotAgent
from agent.schemas import ollama_tools
from agent.tool_text import recover_tool_calls_from_text
from config.settings import (
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_WARMUP_ENABLED,
)
from config.sites import SiteProfile
from robot.base import RobotDriver

OLLAMA_AGENT_HINT = (
    "You control a real robot via tools. "
    "Always invoke tools through the tool-calling API — never write move_left(0.05) as plain text. "
    "Use get_robot_state before any motion. "
    "Left/right use move_left and move_right (base Y); up/down use move_up and move_down (base Z). "
    "You may call multiple tools across turns until the task is done. "
    "If the user repeats a move request, execute the motion tool again every time. "
    "You cannot call move_home, move_joint, release_rtde_control, or run_urp_program. "
    "On errors, stop and explain — never try to home the robot. "
    "After two failed motion tools, stop and report — do not keep trying smaller moves. "
    "Be brief: one short sentence before each tool call; minimal final summary."
)


def ollama_runtime_options(num_predict: int | None = None) -> dict:
    opts: dict = {"num_predict": num_predict if num_predict is not None else OLLAMA_NUM_PREDICT}
    if OLLAMA_NUM_CTX > 0:
        opts["num_ctx"] = OLLAMA_NUM_CTX
    return opts


def _keep_alive_value():
    raw = (OLLAMA_KEEP_ALIVE or "").strip()
    if not raw or raw.lower() in ("0", "false", "off", "none"):
        return None
    if raw == "-1":
        return -1
    return raw


def warmup_ollama(model: str | None = None) -> str:
    """Load model into Ollama so the first Agentic AI goal is faster."""
    if not OLLAMA_WARMUP_ENABLED:
        return "disabled (OLLAMA_WARMUP_ENABLED=0)"
    model = model or OLLAMA_MODEL
    client = Client(host=OLLAMA_HOST)
    t0 = time.time()
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "options": ollama_runtime_options(num_predict=8),
    }
    keep_alive = _keep_alive_value()
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive
    client.chat(**kwargs)
    return f"{model} ready in {time.time() - t0:.1f}s (keep_alive={OLLAMA_KEEP_ALIVE})"


class OllamaRobotAgent(BaseRobotAgent):
    def __init__(self, robot: RobotDriver, site: SiteProfile, model: str | None = None):
        super().__init__(robot, site)
        self.model = model or OLLAMA_MODEL
        self.tools = ollama_tools()
        self.client = Client(host=OLLAMA_HOST)
        self._last_message = None

    def _provider_label(self) -> str:
        return f"ollama/{self.model}"

    def _initial_messages(self, goal: str) -> list:
        return [
            {"role": "system", "content": f"{self.system_prompt}\n\n{OLLAMA_AGENT_HINT}"},
            {"role": "user", "content": goal},
        ]

    def _llm_step(self, messages: list) -> tuple[str | None, list, bool]:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools,
            "options": ollama_runtime_options(),
        }
        keep_alive = _keep_alive_value()
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        response = self.client.chat(**kwargs)
        msg = response.message
        self._last_message = msg
        text = (msg.content or "").strip() or None
        tool_calls = self._parse_tool_calls(msg)
        if not tool_calls and text:
            tool_calls = recover_tool_calls_from_text(text)
            if tool_calls:
                names = ", ".join(c["name"] for c in tool_calls)
                print(f"  [ollama] recovered tool call(s) from text: {names}")
        is_final = not tool_calls
        return text, tool_calls, is_final

    @staticmethod
    def _parse_tool_calls(msg) -> list[dict]:
        if not msg.tool_calls:
            return []
        out = []
        for tc in msg.tool_calls:
            fn = tc.function
            args = fn.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {}
            elif args is None:
                args = {}
            out.append({"name": fn.name, "arguments": args})
        return out

    def _append_tool_round(self, messages: list, tool_calls: list) -> tuple[list, str | None]:
        for call in tool_calls:
            self._log(f"TOOL: {call['name']} {call['arguments']}")

        results, note = self._execute_tools(self.robot, self.policy, tool_calls)

        new_messages = list(messages)
        if self._last_message is not None:
            new_messages.append(self._last_message)
        for call, res in zip(tool_calls, results):
            new_messages.append(
                {
                    "role": "tool",
                    "tool_name": call["name"],
                    "content": json.dumps(res["result"]),
                }
            )
            self._log(f"RESULT: {res['result']}")

        return new_messages, note


def _model_available(listed: list[str], model: str) -> bool:
    if model in listed:
        return True
    base = model.split(":")[0]
    return any(m == base or m.startswith(f"{base}:") for m in listed)


def check_ollama_ready(model: str | None = None) -> None:
    """Raise with helpful message if Ollama is down or model missing."""
    model = model or OLLAMA_MODEL
    try:
        client = Client(host=OLLAMA_HOST)
        listed = [m.model for m in client.list().models]
        if not _model_available(listed, model):
            raise RuntimeError(
                f"Model '{model}' not found. Pull it:\n  ollama pull {model}\n"
                f"Installed: {listed or '(none)'}"
            )
    except RuntimeError:
        raise
    except Exception as e:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Start it: ollama serve\n  ({e})"
        ) from e
