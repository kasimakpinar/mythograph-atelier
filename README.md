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

## Development With Codex

This project was developed with OpenAI Codex as a local coding collaborator. The submitted app is designed not to call OpenAI APIs at runtime.
