# agent/ollama_agent.py — local Ollama agentic loop with tool calling

import json
import os

from ollama import Client

from agent.base_agent import BaseRobotAgent
from agent.schemas import ollama_tools
from config.settings import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_NUM_PREDICT
from config.sites import SiteProfile
from robot.base import RobotDriver

OLLAMA_AGENT_HINT = (
    "You control a real robot via tools. "
    "Use get_robot_state before any motion. "
    "You may call multiple tools across turns until the task is done."
)


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
        response = self.client.chat(
            model=self.model,
            messages=messages,
            tools=self.tools,
            options={"num_predict": OLLAMA_NUM_PREDICT},
        )
        msg = response.message
        self._last_message = msg
        text = (msg.content or "").strip() or None
        tool_calls = self._parse_tool_calls(msg)
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

    def _append_tool_round(self, messages: list, tool_calls: list) -> list:
        for call in tool_calls:
            self._log(f"TOOL: {call['name']} {call['arguments']}")

        results = self._execute_tools(self.robot, self.policy, tool_calls)

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

        return new_messages


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
