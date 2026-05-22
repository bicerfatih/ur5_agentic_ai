#!/usr/bin/env python3
"""Verify Ollama is running and the configured model is available."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ollama_agent import check_ollama_ready
from config.settings import OLLAMA_HOST, OLLAMA_MODEL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=OLLAMA_MODEL)
    args = p.parse_args()

    print(f"Ollama host: {OLLAMA_HOST}")
    print(f"Model:       {args.model}")

    try:
        check_ollama_ready(args.model)
        print("\nPASS: Ollama is ready for agentic control.")
        print(f"\nPull a tool-capable model if needed:")
        print(f"  ollama pull {args.model}")
        print("\nRun agent (dry-run):")
        print("  python3 main.py --dry-run --llm ollama")
        sys.exit(0)
    except (ConnectionError, RuntimeError) as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
