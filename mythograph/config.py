from pathlib import Path


APP_TITLE = "Mythograph Atelier"
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TRACE_PATH = DATA_DIR / "traces.jsonl"
OUTPUT_DIR = ROOT_DIR / "outputs"
