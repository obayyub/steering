#!/bin/bash
# Setup script for Lambda Labs
# Run: curl -sSL <raw-github-url> | bash
# Or: bash setup_lambda.sh

set -e

echo "=== Setting up steering experiment ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Clone repo (update with your repo URL)
if [ ! -d "steering" ]; then
    echo "Cloning repo..."
    git clone https://github.com/YOUR_USERNAME/steering.git
fi

cd steering

# Install dependencies
echo "Installing dependencies..."
uv sync

# Verify GPU
echo "Checking GPU..."
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run all models:"
echo "  uv run python run_all_models.py"
echo ""
echo "To run specific sizes:"
echo "  uv run python run_all_models.py --models 0.6B 1.7B 4B"
echo ""
echo "Results will be saved to results/"
