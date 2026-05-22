# agent/claude_agent.py — Anthropic Claude agentic loop (optional backend)

import json

import anthropic

from agent.base_agent import BaseRobotAgent
from config.settings import CLAUDE_MAX_TOKENS, CLAUDE_MODEL
from config.sites import SiteProfile
from robot.base import RobotDriver
from robot.tools import TOOL_SCHEMAS


class ClaudeRobotAgent(BaseRobotAgent):
    def __init__(self, robot: RobotDriver, site: SiteProfile):
        super().__init__(robot, site)
        self.client = anthropic.Anthropic()
        self._pending_assistant_blocks = []

    def _provider_label(self) -> str:
        return f"claude/{CLAUDE_MODEL}"

    def _initial_messages(self, goal: str) -> list:
        return [{"role": "user", "content": goal}]

    def _llm_step(self, messages: list) -> tuple[str | None, list, bool]:
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=self.system_prompt,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        texts = []
        tool_calls = []
        assistant_blocks = []
        for block in response.content:
            assistant_blocks.append(block)
            if hasattr(block, "text") and block.text:
                texts.append(block.text)
            if block.type == "tool_use":
                tool_calls.append({"name": block.name, "arguments": block.input})

        self._pending_assistant_blocks = assistant_blocks
        text = "\n".join(texts).strip() or None
        is_final = response.stop_reason == "end_turn"
        return text, tool_calls, is_final

    def _append_tool_round(self, messages: list, tool_calls: list) -> list:
        for call in tool_calls:
            self._log(f"TOOL: {call['name']} {call['arguments']}")

        results = self._execute_tools(self.robot, self.policy, tool_calls)

        tool_results = []
        tool_blocks = [b for b in self._pending_assistant_blocks if b.type == "tool_use"]
        for block, res in zip(tool_blocks, results):
            self._log(f"RESULT: {res['result']}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(res["result"]),
                }
            )

        return messages + [
            {"role": "assistant", "content": self._pending_assistant_blocks},
            {"role": "user", "content": tool_results},
        ]
