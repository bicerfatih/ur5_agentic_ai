#!/bin/bash
# Install Python deps for the agent (run from ur5_agent/)
set -e
cd "$(dirname "$0")/.."

if [ -d robot_env ]; then
  source robot_env/bin/activate
  echo "Using robot_env"
else
  echo "No robot_env — using system python3 (consider: bash setup.sh)"
fi

pip install --upgrade pip
pip install -r requirements.txt
python3 -c "from ollama import Client; print('OK: ollama Python package')"
python3 scripts/check_ollama.py
