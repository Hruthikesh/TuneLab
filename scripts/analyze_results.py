#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to pythonpath if running directly as a script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from tunelab.analysis.cli import main

if __name__ == "__main__":
    main()
