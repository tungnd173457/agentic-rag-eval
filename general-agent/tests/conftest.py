"""Cho pytest import được config/ingestion/... của bản copy general-agent."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # general-agent/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
