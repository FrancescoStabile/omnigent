"""Omnigent test configuration."""

import sys
from pathlib import Path

# Ensure omnigent is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
