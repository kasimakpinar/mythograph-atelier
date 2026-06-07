---
title: Mythograph Atelier
emoji: 🎨
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
short_description: AI abstract art with personal meaning
---

# Mythograph Atelier

Abstract art with a meaning you can explain.

Mythograph Atelier is a Gradio app for Hugging Face's Build Small Hackathon. It creates an abstract painting backwards: first a small local text model asks short adaptive questions and chooses safe UI controls, then FLUX.2 Klein renders a landscape abstract artwork with a title, symbol map, and short explanation.

## Current MVP

- Chat-first dynamic interview with starter chips, assistant messages, and controls that appear only when needed.
- No external inference APIs: `llama.cpp` runs the text model on ZeroGPU, and FLUX.2 Klein runs afterward on ZeroGPU for image generation.
- Dynamic control tray for choice cards, multi-choice cards, visual sliders, palette mood, text refinement, and create readiness.
- FLUX.2 Klein image generation with Pillow as a reliable fallback.
- JSONL trace logging for conversation turns, control responses, model calls, image generation, and demo sessions.
- Custom Gradio shell designed for the Off-Brand badge path.

## Hackathon Fit

- **Track:** An Adventure in Thousand Token Wood
- **Small models only:** Nemotron-3-Nano-4B for text and FLUX.2 Klein 4B for images.
- **Built on Gradio:** The app is a Gradio Blocks application.
- **Off the Grid:** No cloud inference APIs are required.
- **Llama Champion:** Text inference runs through `llama.cpp`.

## Run Locally

For quick UI testing without model download:

```bash
MYTHOGRAPH_LLM_MODE=mock python app.py
```

For the no-API llama.cpp path:

```bash
pip install -r requirements.txt
python app.py
```

## HF Space Configuration

The intended public Space configuration is:

```text
MYTHOGRAPH_LLM_MODE=llamacpp
MYTHOGRAPH_LLAMACPP_REPO_ID=nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF
MYTHOGRAPH_LLAMACPP_FILENAME=NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf
MYTHOGRAPH_LLAMACPP_N_CTX=2048
MYTHOGRAPH_LLAMACPP_N_GPU_LAYERS=-1
MYTHOGRAPH_LLAMACPP_N_THREADS=2
MYTHOGRAPH_LLAMACPP_PRELOAD=0
MYTHOGRAPH_LLAMACPP_CHAT_ENABLED=1
MYTHOGRAPH_LLAMACPP_RECIPE_ENABLED=1
MYTHOGRAPH_LLAMACPP_UNLOAD_AFTER_CALL=0
MYTHOGRAPH_LLAMACPP_FLASH_ATTN=0
MYTHOGRAPH_LLM_CHAT_MAX_TOKENS=220
MYTHOGRAPH_LLM_RECIPE_MAX_TOKENS=220
MYTHOGRAPH_LLM_TEMPERATURE=0.55
MYTHOGRAPH_CONVERSATION_MODE=model_assisted
MYTHOGRAPH_IMAGE_MODE=flux
MYTHOGRAPH_IMAGE_MODEL_ID=black-forest-labs/FLUX.2-klein-4B
MYTHOGRAPH_IMAGE_WIDTH=1024
MYTHOGRAPH_IMAGE_HEIGHT=768
MYTHOGRAPH_IMAGE_STEPS=8
MYTHOGRAPH_IMAGE_DTYPE=float16
MYTHOGRAPH_IMAGE_CPU_OFFLOAD=1
```

The default MVP path uses short GPU llama.cpp calls for creative chat turns and the final recipe. Before FLUX renders, the app unloads llama.cpp and clears CUDA memory so image generation can own the next ZeroGPU allocation.

## Model Architecture

Text:

```python
Llama.from_pretrained(
    repo_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
    filename="NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=2,
    n_gpu_layers=-1,
)
```

The model-assisted conversation director receives a compact atelier state, not the full chat transcript. It returns one safe JSON `ConversationTurn`; Python validates the component kind, options, sliders, and readiness before updating the UI.

Image:

```text
FLUX.2 Klein 4B
1024x768 landscape
8 steps
ZeroGPU after llama.cpp unload
```

If FLUX fails to load or generate, the app falls back to Pillow and records the failure in the trace.

## Trace Proof

For a successful no-API GPU run, the downloaded trace should show:

```text
llm_conversation_turn.source = llamacpp
llm_conversation_turn.used_fallback = false
llm_art_recipe.source = llamacpp
llm_art_recipe.used_fallback = false
image_generation.source = flux_klein
```

## Development With Codex

This project was developed with OpenAI Codex as a local coding collaborator. The submitted app is designed not to call OpenAI APIs or external inference APIs at runtime.
