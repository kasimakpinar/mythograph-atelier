import os
import subprocess

import modal


APP_NAME = "mythograph-nemotron"
MODEL_ID = os.getenv("MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "nemotron-nano-9b-v2")
GPU_TYPE = os.getenv("MODAL_GPU_TYPE", "L40S")
VLLM_PORT = 8000


app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.15.1", add_python="3.12")
    .pip_install("huggingface_hub[hf_transfer]>=0.34.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 30,
    scaledown_window=60 * 5,
    secrets=[modal.Secret.from_name("mythograph-vllm")],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=60 * 20)
def serve():
    api_key = os.environ["VLLM_API_KEY"]
    command = [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--api-key",
        api_key,
        "--dtype",
        "auto",
        "--max-model-len",
        os.getenv("MAX_MODEL_LEN", "8192"),
    ]
    subprocess.Popen(command)
