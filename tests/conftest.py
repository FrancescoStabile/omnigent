"""Shared test fixtures for Omnigent."""

import pytest
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
