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

Mythograph Atelier is a Gradio app for Hugging Face's Build Small Hackathon. It creates an abstract painting backwards: first it learns a small amount about the ideas, symbols, and visual taste a user connects with, then it creates an artwork with a title, symbol map, and short "friend explanation."

## Current MVP

- Controlled dynamic interview actions: idea cards, visual sliders, style cards, symbol cards, contrast questions, and surprise steps.
- Hidden objective scoring to decide what the interface should ask next.
- Offline procedural artwork generation with Pillow as a reliable fallback.
- Final gallery card with painting, title, friend explanation, symbol map, image prompt, and regeneration controls.
- JSONL trace logging for demo sessions.
- Custom Gradio CSS for a more atelier-like interface.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Hackathon Fit

- **Track:** An Adventure in Thousand Token Wood
- **Small models only:** The current MVP uses no external model at runtime. Planned model integrations are designed for llama.cpp-compatible text models and small image models.
- **Built on Gradio:** The app is a Gradio Blocks application.
- **Off the Grid:** The fallback flow runs locally without cloud APIs.

## Planned Model Integrations

- Text/art direction: a <=32B model through llama.cpp or another local runtime.
- Image generation: FLUX.2-klein or another hackathon-eligible small image model.
- Fine-tuned art director: optional LoRA/SFT model for structured art recipes.

## Image Generation

The image layer is now routed through an `ImageClient`, but the active public Space backend is still the local Pillow renderer:

```bash
MYTHOGRAPH_IMAGE_MODE=pillow
```

Trace exports include an `image_generation` event with `source`, `elapsed_seconds`, `error`, and `image_path`. Today the expected source is `pillow_fallback`. A future FLUX or other small image model can be added behind the same interface without changing the UI flow.

Optional FLUX.2-klein image backend:

```bash
MYTHOGRAPH_IMAGE_MODE=flux
MYTHOGRAPH_IMAGE_MODEL_ID=black-forest-labs/FLUX.2-klein-4B
MYTHOGRAPH_IMAGE_WIDTH=1024
MYTHOGRAPH_IMAGE_HEIGHT=1024
MYTHOGRAPH_IMAGE_STEPS=8
MYTHOGRAPH_IMAGE_DTYPE=float16
```

HF Spaces only installs `requirements.txt` automatically. When we are ready to test FLUX, Codex should temporarily move the dependencies from `requirements-image.txt` into `requirements.txt`, commit, and push. Use it only when testing this backend on suitable GPU hardware. If FLUX fails to load or generate, the app falls back to Pillow and records the error in the `image_generation` trace event.

FLUX.2-klein does not currently use a `negative_prompt` argument in this app path. The prompt itself still asks for no text, letters, signatures, or watermarks. The default image dtype is `float16`; set `MYTHOGRAPH_IMAGE_DTYPE=bfloat16` only if the selected hardware/runtime supports it cleanly.

## Local LLM Configuration

The app defaults to safe mock mode so the Space remains usable even when no model server is running.

```bash
MYTHOGRAPH_LLM_MODE=mock python app.py
```

To test with a small local llama.cpp/OpenAI-compatible model:

```bash
MYTHOGRAPH_LLM_MODE=local \
MYTHOGRAPH_LLM_BASE_URL=http://127.0.0.1:8080/v1 \
MYTHOGRAPH_LLM_MODEL=tiny-local-test-model \
python app.py
```

For the final Space, the same variables can point at a larger hackathon-eligible model, as long as total model parameters stay at or below 32B.

By default, the dynamic interview uses the deterministic fast path even when model mode is enabled. This keeps each question responsive while testing llama.cpp on the final Create step.

```bash
MYTHOGRAPH_MODEL_UI_DIRECTOR=0
```

Set `MYTHOGRAPH_MODEL_UI_DIRECTOR=1` only when testing a fast enough text model for every interview turn.

Target model for the text/art director layer:

```bash
MYTHOGRAPH_LLM_MODE=local
MYTHOGRAPH_LLM_BASE_URL=http://127.0.0.1:8000/v1
MYTHOGRAPH_LLM_MODEL=nvidia/OpenReasoning-Nemotron-32B
```

The current public Space intentionally defaults to mock mode until a model server is added to the Space runtime or attached as a local OpenAI-compatible service.

## llama.cpp Badge Path

The app also supports an optional in-process llama.cpp runtime through `llama-cpp-python`.

Keep the public Space in mock mode until you are ready to spend credits. For the badge path, first test with a tiny GGUF, then switch to Nemotron:

```bash
MYTHOGRAPH_LLM_MODE=llamacpp
MYTHOGRAPH_LLAMACPP_REPO_ID=lmstudio-community/Qwen3.5-0.8B-GGUF
MYTHOGRAPH_LLAMACPP_FILENAME=Qwen3.5-0.8B-Q4_K_M.gguf
MYTHOGRAPH_LLAMACPP_N_CTX=2048
MYTHOGRAPH_LLAMACPP_N_GPU_LAYERS=-1
MYTHOGRAPH_MODEL_UI_DIRECTOR=0
```

Nemotron demo settings:

```bash
MYTHOGRAPH_LLM_MODE=llamacpp
MYTHOGRAPH_LLAMACPP_REPO_ID=Triangle104/OpenReasoning-Nemotron-32B-Q4_K_M-GGUF
MYTHOGRAPH_LLAMACPP_FILENAME=*q4_k_m.gguf
MYTHOGRAPH_LLAMACPP_N_CTX=4096
MYTHOGRAPH_LLAMACPP_N_GPU_LAYERS=-1
```

Install the optional runtime dependencies from `requirements-llamacpp.txt` only when enabling this mode.

For the public Space proof run, `requirements.txt` already includes the CPU prebuilt `llama-cpp-python` wheel index. This lets the Space import llama.cpp without downloading a model until the runtime mode is changed.

HF Space proof steps:

1. Let the Space rebuild after this commit.
2. Keep hardware on CPU/ZeroGPU for the tiny proof.
3. Add Space variables:

```text
MYTHOGRAPH_LLM_MODE=llamacpp
MYTHOGRAPH_LLAMACPP_REPO_ID=lmstudio-community/Qwen3.5-0.8B-GGUF
MYTHOGRAPH_LLAMACPP_FILENAME=Qwen3.5-0.8B-Q4_K_M.gguf
MYTHOGRAPH_LLAMACPP_N_CTX=2048
MYTHOGRAPH_LLAMACPP_N_GPU_LAYERS=-1
```

4. Restart the Space.
5. Confirm the UI shows `Runtime: llamacpp`.
6. Generate once and download the trace. The trace should include `source: llamacpp`.

Cost guardrails:

- Keep CPU/ZeroGPU mock mode active while polishing the UI.
- Use a tiny GGUF for the first llama.cpp proof.
- Switch paid hardware on only for recorded testing and the final demo window.
- Stop or downgrade paid hardware immediately after recording.

As of the current Hugging Face docs, ZeroGPU over-quota usage is billed at $1 per 10 GPU minutes, while paid Spaces hardware is hourly. A single L40S is listed at $1.80/hour and a single A100 at $2.50/hour, so the $20 credit budget is workable if we keep paid runtime windows short.

## Development With Codex

This project was developed with OpenAI Codex as a local coding collaborator. The submitted app is designed not to call OpenAI APIs at runtime.
