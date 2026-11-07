import sys
from pathlib import Path

MODEL1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL1_DIR))
sys.path.insert(0, str(MODEL1_DIR / "tools"))
