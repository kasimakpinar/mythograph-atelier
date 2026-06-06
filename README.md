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

- Chat-first dynamic interview with a centered starting prompt, starter chips, assistant messages, and controls that appear only when needed.
- Hybrid conversation director: deterministic routing for fast turns, llama.cpp/Nemotron for the final art recipe.
- Dynamic control tray for choice cards, multi-choice cards, visual sliders, palette mood, text refinement, and create readiness.
- FLUX.2-klein image generation with Pillow as a reliable fallback.
- Final gallery card with painting, title, friend explanation, symbol map, image prompt, and regeneration controls.
- JSONL trace logging for conversation turns, control responses, model calls, image generation, and demo sessions.
- Custom Gradio shell designed for the Off-Brand badge path.

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

The image layer is routed through an `ImageClient`. The current public Space test backend is FLUX.2-klein, with Pillow kept as a reliable fallback:

```bash
MYTHOGRAPH_IMAGE_MODE=flux
```

Trace exports include an `image_generation` event with `source`, `elapsed_seconds`, `error`, and `image_path`. For the FLUX path, the expected source is `flux_klein`; if model loading or generation fails, the app falls back to Pillow and records the error.

Optional FLUX.2-klein image backend:

```bash
MYTHOGRAPH_IMAGE_MODE=flux
MYTHOGRAPH_IMAGE_MODEL_ID=black-forest-labs/FLUX.2-klein-4B
MYTHOGRAPH_IMAGE_WIDTH=1024
MYTHOGRAPH_IMAGE_HEIGHT=1024
MYTHOGRAPH_IMAGE_STEPS=8
MYTHOGRAPH_IMAGE_DTYPE=float16
MYTHOGRAPH_IMAGE_GUIDANCE_SCALE=1.0
MYTHOGRAPH_IMAGE_CPU_OFFLOAD=1
```

HF Spaces only installs `requirements.txt` automatically, so the active Space requirements include both the FLUX dependencies and the llama.cpp proof dependencies. If FLUX fails to load or generate, the app falls back to Pillow and records the error in the `image_generation` trace event.

FLUX.2-klein does not currently use a `negative_prompt` argument in this app path. The prompt itself still asks for no text, letters, signatures, or watermarks. The default image dtype is `float16`; set `MYTHOGRAPH_IMAGE_DTYPE=bfloat16` only if the selected hardware/runtime supports it cleanly. The backend uses `Flux2KleinPipeline` with CPU offload by default to reduce VRAM pressure.

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

Set `MYTHOGRAPH_MODEL_UI_DIRECTOR=1` only when testing the legacy interviewer path. The chat-first UI uses `MYTHOGRAPH_CONVERSATION_MODE` instead.

Hosted OpenAI-compatible text endpoints, including Modal vLLM and Hugging Face Inference Providers, use:

```bash
MYTHOGRAPH_LLM_MODE=local
MYTHOGRAPH_LLM_BASE_URL=https://your-endpoint.example/v1
MYTHOGRAPH_LLM_MODEL=nemotron-nano-9b-v2
MYTHOGRAPH_LLM_API_KEY=your-api-key
MYTHOGRAPH_LLM_SOURCE_LABEL=modal
MYTHOGRAPH_LLM_TIMEOUT_SECONDS=120
MYTHOGRAPH_LLM_MAX_TOKENS=1200
MYTHOGRAPH_LLM_TEMPERATURE=0.7
```

For model-authored interview turns:

```bash
MYTHOGRAPH_CONVERSATION_MODE=model_assisted
```

If the hosted model fails, returns invalid JSON, or tries to show an unsafe UI control, the app falls back to the deterministic conversation turn and records `llm_conversation_turn` in the trace.

Target model for the text/art director layer:

```bash
MYTHOGRAPH_LLM_MODE=local
MYTHOGRAPH_LLM_BASE_URL=http://127.0.0.1:8000/v1
MYTHOGRAPH_LLM_MODEL=nvidia/OpenReasoning-Nemotron-32B
```

The current public Space intentionally defaults to mock mode until a model server is added to the Space runtime or attached as a local OpenAI-compatible service.

## Modal Nemotron Endpoint

This repo includes `modal_vllm_nemotron.py`, which deploys NVIDIA Nemotron Nano 9B v2 behind a vLLM OpenAI-compatible server.

User setup:

```bash
pip install modal
modal setup
modal secret create mythograph-vllm VLLM_API_KEY=choose-a-long-random-string
modal deploy modal_vllm_nemotron.py
```

If the model requires Hugging Face auth in your account, create the same secret with both values instead:

```bash
modal secret create mythograph-vllm VLLM_API_KEY=choose-a-long-random-string HF_TOKEN=your_huggingface_token
```

After deploy, copy the Modal URL and add `/v1` for the Space base URL.

Quick Modal checks:

```bash
curl https://YOUR-MODAL-APP.modal.run/health
curl https://YOUR-MODAL-APP.modal.run/v1/chat/completions \
  -H "Authorization: Bearer YOUR_VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"nemotron-nano-9b-v2\",\"messages\":[{\"role\":\"user\",\"content\":\"Return one short sentence.\"}],\"max_tokens\":40}"
```

HF Space variables for Modal:

```text
MYTHOGRAPH_LLM_MODE=local
MYTHOGRAPH_LLM_BASE_URL=https://YOUR-MODAL-APP.modal.run/v1
MYTHOGRAPH_LLM_MODEL=nemotron-nano-9b-v2
MYTHOGRAPH_LLM_API_KEY=YOUR_VLLM_API_KEY
MYTHOGRAPH_LLM_SOURCE_LABEL=modal
MYTHOGRAPH_LLM_TIMEOUT_SECONDS=120
MYTHOGRAPH_CONVERSATION_MODE=model_assisted
MYTHOGRAPH_MODEL_UI_DIRECTOR=0
MYTHOGRAPH_IMAGE_MODE=flux
```

Later, the same app path can use Hugging Face Inference Providers and HF credits:

```text
MYTHOGRAPH_LLM_MODE=local
MYTHOGRAPH_LLM_BASE_URL=https://router.huggingface.co/v1
MYTHOGRAPH_LLM_MODEL=<provider-supported-model>
MYTHOGRAPH_LLM_API_KEY=<HF token with Inference Providers permission>
MYTHOGRAPH_LLM_SOURCE_LABEL=hf-router
MYTHOGRAPH_CONVERSATION_MODE=model_assisted
```

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

Recommended Nemotron demo settings:

```bash
MYTHOGRAPH_LLM_MODE=llamacpp
MYTHOGRAPH_LLAMACPP_REPO_ID=bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF
MYTHOGRAPH_LLAMACPP_FILENAME=nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf
MYTHOGRAPH_LLAMACPP_N_CTX=2048
MYTHOGRAPH_LLAMACPP_N_GPU_LAYERS=-1
```

Experimental 32B settings, only after the 9B run works:

```bash
MYTHOGRAPH_LLM_MODE=llamacpp
MYTHOGRAPH_LLAMACPP_REPO_ID=bartowski/nvidia_OpenReasoning-Nemotron-32B-GGUF
MYTHOGRAPH_LLAMACPP_FILENAME=nvidia_OpenReasoning-Nemotron-32B-Q4_K_M.gguf
MYTHOGRAPH_LLAMACPP_N_CTX=4096
MYTHOGRAPH_LLAMACPP_N_GPU_LAYERS=-1
```

Install the optional runtime dependencies from `requirements-llamacpp.txt` only when enabling this mode.

For the public Space proof run, `requirements.txt` already includes the CUDA prebuilt `llama-cpp-python` wheel index and CUDA runtime packages. The app also preloads CUDA libraries from pip's `site-packages/nvidia/...` folders before importing `llama_cpp`, because some Space images do not expose those libraries on the dynamic loader path by default.

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
