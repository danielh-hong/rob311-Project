#!/usr/bin/env python3
"""
Cross-platform launcher for Bazaar AI
Works on Windows, Mac, and Linux
"""

import subprocess
import sys
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()

# Navigate to the UI directory
ui_dir = script_dir / "src" / "bazaar-ai" / "ui"
launch_script = ui_dir / "launch.py"

if not launch_script.exists():
    print(f"Error: Could not find launch.py at {launch_script}")
    sys.exit(1)

# Run the launch script
try:
    subprocess.run([sys.executable, str(launch_script)], cwd=str(ui_dir))
except KeyboardInterrupt:
    print("\n\nShutdown complete.")
except Exception as e:
    print(f"Error launching Bazaar AI: {e}")
    sys.exit(1)
