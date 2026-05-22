#!/bin/bash
echo "Setting up UR5 Agentic AI environment..."

# Install system dependencies
sudo apt install -y python3-full python3-pip cmake build-essential libboost-all-dev python3-dev

# Create virtual environment
python3 -m venv robot_env
source robot_env/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. source robot_env/bin/activate"
echo "  2. ollama pull qwen2.5:7b   # if not already pulled"
echo "  3. python3 scripts/check_ollama.py"
echo "  4. python3 main.py --dry-run --site lab"
echo ""
echo "Note: Ollama server runs via systemd — do not run 'ollama serve' again."
