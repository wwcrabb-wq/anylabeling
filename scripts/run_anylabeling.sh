#!/bin/bash
# Simple script to run AnyLabeling application
# This script will activate a virtual environment if present, or run directly

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check for virtual environment and activate if present
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Run the application
echo "Starting AnyLabeling..."
python anylabeling/app.py "$@"
