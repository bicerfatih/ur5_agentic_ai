# agent/factory.py — create LLM backend for robot agent

import os

from agent.ollama_agent import OllamaRobotAgent, check_ollama_ready
from config.sites import SiteProfile
from robot.base import RobotDriver


def create_agent(
    robot: RobotDriver,
    site: SiteProfile,
    llm: str | None = None,
    ollama_model: str | None = None,
):
    backend = (llm or os.environ.get("LLM_BACKEND", "ollama")).lower()

    if backend in ("ollama", "local"):
        check_ollama_ready(ollama_model)
        return OllamaRobotAgent(robot, site, model=ollama_model)

    if backend in ("claude", "anthropic"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Use --llm ollama (default) or export your key."
            )
        from agent.claude_agent import ClaudeRobotAgent

        return ClaudeRobotAgent(robot, site)

    raise ValueError(f"Unknown LLM backend: {backend}. Use ollama or claude.")
