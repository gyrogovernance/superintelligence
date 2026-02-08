# GyroGem/manual_check.py
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from GyroGem.agent.model import GyroGemModel

model = GyroGemModel()
text = "The AI agent decides which users should be banned."

expr = model.classify(text)
print("Input:", text)
print("THM:", expr)