# agent/base_agent.py — shared agentic loop (policy, logging, tool execution)

import datetime
import json
import os

from agent.prompts import build_system_prompt
from config.settings import LOG_FILE
from config.sites import SiteProfile
from policy.safety import PolicyEngine
from robot.base import RobotDriver


class BaseRobotAgent:
    """Shared run loop; subclasses implement LLM provider calls."""

    def __init__(self, robot: RobotDriver, site: SiteProfile):
        self.robot = robot
        self.site = site
        self.policy = PolicyEngine(site=site)
        self.system_prompt = build_system_prompt(site, robot)
        self.last_run_note: str | None = None
        os.makedirs("logs", exist_ok=True)

    def _log(self, text: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] [{self.site.site_id}] {text}\n")

    def _provider_label(self) -> str:
        return "agent"

    def _llm_step(self, messages: list) -> tuple[str | None, list, bool]:
        """
        One LLM turn.
        Returns (assistant_text, tool_calls, is_final_turn).
        tool_calls: list of dicts with keys name, arguments (dict).
        """
        raise NotImplementedError

    def run(self, goal: str):
        print(f"\n{'━' * 55}")
        print(f"  GOAL: {goal}")
        print(f"  SITE: {self.site.display_name}")
        print(f"  LLM:  {self._provider_label()}")
        print(f"{'━' * 55}\n")
        self._log(f"GOAL: {goal}")

        self.policy.begin_goal()
        self.last_run_note = None
        messages = self._initial_messages(goal)
        step = 0
        max_steps = self.site.max_steps_per_goal

        while True:
            step += 1
            print(f"⏳  LLM step {step} (first call may take 1–3 min on Jetson)...", flush=True)
            try:
                text, tool_calls, is_final = self._llm_step(messages)
            except Exception as e:
                print(f"❌ LLM error: {e}")
                self._log(f"LLM ERROR: {e}")
                break

            if text:
                print(f"🤖  {text}")
                self._log(f"AGENT: {text}")

            if is_final or not tool_calls:
                print(f"\n✅  Task complete after {step} steps.")
                self._log(f"DONE after {step} steps")
                break

            messages, notes = self._append_tool_round(messages, tool_calls)
            if notes:
                self.last_run_note = notes

            if step >= max_steps:
                print(f"⚠️  Max steps ({max_steps}) reached. Stopping.")
                self._log("WARN: max steps reached")
                break

    def _initial_messages(self, goal: str) -> list:
        raise NotImplementedError

    def _append_tool_round(self, messages: list, tool_calls: list) -> tuple[list, str | None]:
        raise NotImplementedError

    @staticmethod
    def _execute_tools(robot, policy, tool_calls: list) -> tuple[list[dict], str | None]:
        from robot.tools import execute_tool

        results = []
        last_note = None
        for call in tool_calls:
            name = call["name"]
            args = call.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args) if args else {}

            print(f"\n🔧  {name}({json.dumps(args, separators=(',', ':'))})")

            try:
                result = execute_tool(name, args, robot, policy, caller="agent")
            except ValueError as e:
                result = {"status": "error", "reason": str(e)}
                print(f"   ❌ Tool error: {e}")
            except Exception as e:
                result = {"status": "error", "reason": str(e)}
                print(f"   ❌ Tool error: {e}")

            print(f"   → {result}")
            results.append({"name": name, "result": result})
            if isinstance(result, dict) and result.get("status") == "error":
                last_note = f"{name}: {result.get('reason', 'error')}"
        return results, last_note
