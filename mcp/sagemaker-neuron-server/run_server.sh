#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"
if [ ! -d "$VENV" ] || ! "$VENV/bin/python" -c "import sagemaker_neuron_server" 2>/dev/null; then
    rm -rf "$VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install "$DIR" 2>/dev/null
fi
exec "$VENV/bin/python" -m sagemaker_neuron_server
