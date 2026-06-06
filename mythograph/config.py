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
MODEL_UI_DIRECTOR_ENABLED = os.getenv("MYTHOGRAPH_MODEL_UI_DIRECTOR", "0").strip() == "1"

LLAMACPP_REPO_ID = os.getenv("MYTHOGRAPH_LLAMACPP_REPO_ID", "lmstudio-community/Qwen3.5-0.8B-GGUF")
LLAMACPP_FILENAME = os.getenv("MYTHOGRAPH_LLAMACPP_FILENAME", "Qwen3.5-0.8B-Q4_K_M.gguf")
LLAMACPP_N_CTX = int(os.getenv("MYTHOGRAPH_LLAMACPP_N_CTX", "2048"))
LLAMACPP_N_GPU_LAYERS = int(os.getenv("MYTHOGRAPH_LLAMACPP_N_GPU_LAYERS", "-1"))
LLAMACPP_CHAT_FORMAT = os.getenv("MYTHOGRAPH_LLAMACPP_CHAT_FORMAT", "").strip() or None
LLAMACPP_VERBOSE = os.getenv("MYTHOGRAPH_LLAMACPP_VERBOSE", "0").strip() == "1"

IMAGE_MODE = os.getenv("MYTHOGRAPH_IMAGE_MODE", "pillow").strip().lower()
IMAGE_MODEL_ID = os.getenv("MYTHOGRAPH_IMAGE_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
IMAGE_WIDTH = int(os.getenv("MYTHOGRAPH_IMAGE_WIDTH", "1024"))
IMAGE_HEIGHT = int(os.getenv("MYTHOGRAPH_IMAGE_HEIGHT", "1024"))
IMAGE_STEPS = int(os.getenv("MYTHOGRAPH_IMAGE_STEPS", "8"))
IMAGE_SEED = int(os.getenv("MYTHOGRAPH_IMAGE_SEED", "0"))
