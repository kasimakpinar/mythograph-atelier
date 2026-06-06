from pathlib import Path
import os


APP_TITLE = "Mythograph Atelier"
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TRACE_PATH = DATA_DIR / "traces.jsonl"
OUTPUT_DIR = ROOT_DIR / "outputs"

LLM_MODE = os.getenv("MYTHOGRAPH_LLM_MODE", "mock").strip().lower()
LLM_BASE_URL = os.getenv("MYTHOGRAPH_LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LLM_MODEL = os.getenv("MYTHOGRAPH_LLM_MODEL", "local-art-director")
LLM_TIMEOUT_SECONDS = float(os.getenv("MYTHOGRAPH_LLM_TIMEOUT_SECONDS", "30"))
