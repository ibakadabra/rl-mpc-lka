"""Make the repo root importable so `import src.<pkg>` works without install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
