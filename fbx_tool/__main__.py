"""
CLI entry point for FBX Tool.

Enables: python -m fbx_tool
"""

import sys
from fbx_tool.gui.main_window import main

if __name__ == "__main__":
    sys.exit(main())
