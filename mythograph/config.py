from pathlib import Path
import os


APP_TITLE = "Mythograph Atelier"
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TRACE_PATH = DATA_DIR / "traces.jsonl"
OUTPUT_DIR = ROOT_DIR / "outputs"

LLM_MODE = os.getenv("MYTHOGRAPH_LLM_MODE", "llamacpp").strip().lower()
LLM_BASE_URL = os.getenv("MYTHOGRAPH_LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LLM_MODEL = os.getenv("MYTHOGRAPH_LLM_MODEL", "local-dev-model")
LLM_CHAT_MAX_TOKENS = int(os.getenv("MYTHOGRAPH_LLM_CHAT_MAX_TOKENS", "140"))
LLM_RECIPE_MAX_TOKENS = int(os.getenv("MYTHOGRAPH_LLM_RECIPE_MAX_TOKENS", "360"))
LLM_TEMPERATURE = float(os.getenv("MYTHOGRAPH_LLM_TEMPERATURE", "0.55"))
LLM_TIMEOUT_SECONDS = float(os.getenv("MYTHOGRAPH_LLM_TIMEOUT_SECONDS", "30"))
MODEL_UI_DIRECTOR_ENABLED = os.getenv("MYTHOGRAPH_MODEL_UI_DIRECTOR", "0").strip() == "1"
CONVERSATION_MODE = os.getenv("MYTHOGRAPH_CONVERSATION_MODE", "model_assisted").strip().lower()

LLAMACPP_REPO_ID = os.getenv("MYTHOGRAPH_LLAMACPP_REPO_ID", "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF")
LLAMACPP_FILENAME = os.getenv("MYTHOGRAPH_LLAMACPP_FILENAME", "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf")
LLAMACPP_N_CTX = int(os.getenv("MYTHOGRAPH_LLAMACPP_N_CTX", "2048"))
LLAMACPP_N_GPU_LAYERS = int(os.getenv("MYTHOGRAPH_LLAMACPP_N_GPU_LAYERS", "0"))
LLAMACPP_N_THREADS = int(os.getenv("MYTHOGRAPH_LLAMACPP_N_THREADS", "2"))
LLAMACPP_PRELOAD = os.getenv("MYTHOGRAPH_LLAMACPP_PRELOAD", "0").strip() == "1"
LLAMACPP_CHAT_FORMAT = os.getenv("MYTHOGRAPH_LLAMACPP_CHAT_FORMAT", "").strip() or None
LLAMACPP_VERBOSE = os.getenv("MYTHOGRAPH_LLAMACPP_VERBOSE", "0").strip() == "1"

IMAGE_MODE = os.getenv("MYTHOGRAPH_IMAGE_MODE", "flux").strip().lower()
IMAGE_MODEL_ID = os.getenv("MYTHOGRAPH_IMAGE_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
IMAGE_WIDTH = int(os.getenv("MYTHOGRAPH_IMAGE_WIDTH", "1024"))
IMAGE_HEIGHT = int(os.getenv("MYTHOGRAPH_IMAGE_HEIGHT", "768"))
IMAGE_STEPS = int(os.getenv("MYTHOGRAPH_IMAGE_STEPS", "8"))
IMAGE_SEED = int(os.getenv("MYTHOGRAPH_IMAGE_SEED", "0"))
IMAGE_DTYPE = os.getenv("MYTHOGRAPH_IMAGE_DTYPE", "float16").strip().lower()
IMAGE_GUIDANCE_SCALE = float(os.getenv("MYTHOGRAPH_IMAGE_GUIDANCE_SCALE", "1.0"))
IMAGE_CPU_OFFLOAD = os.getenv("MYTHOGRAPH_IMAGE_CPU_OFFLOAD", "1").strip() == "1"
