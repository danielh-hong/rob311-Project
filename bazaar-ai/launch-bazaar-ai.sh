#!/bin/bash
# Launch script for Bazaar AI (Mac/Linux)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/src/bazaar-ai/ui"
python3 launch.py
