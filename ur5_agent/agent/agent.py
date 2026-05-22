# agent/agent.py — backward-compatible exports

from agent.factory import create_agent
from agent.ollama_agent import OllamaRobotAgent
from agent.claude_agent import ClaudeRobotAgent

# Default entry: factory picks backend from LLM_BACKEND / --llm
RobotAgent = OllamaRobotAgent

__all__ = ["create_agent", "RobotAgent", "OllamaRobotAgent", "ClaudeRobotAgent"]
