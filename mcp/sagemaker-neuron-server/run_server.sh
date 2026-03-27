#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -e "$DIR" 2>/dev/null
fi
exec "$VENV/bin/python" -m sagemaker_neuron_server
