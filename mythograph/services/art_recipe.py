import time

from mythograph.schemas.art_recipe import ArtRecipe
from mythograph.schemas.profile import InterviewProfile
from mythograph.config import LLAMACPP_RECIPE_THINKING, LLM_RECIPE_MAX_TOKENS, ROOT_DIR
from mythograph.models.llm_client import LLMClient, extract_json_object
from mythograph.services.trace_logger import log_event

def build_art_recipe_with_model(
    profile: InterviewProfile,
    regeneration_instruction: str | None = None,
    client: LLMClient | None = None,
) -> ArtRecipe:
    llm = client or LLMClient()

    system_prompt = (ROOT_DIR / "mythograph" / "prompts" / "art_recipe_system.txt").read_text(encoding="utf-8")
    started = time.perf_counter()
    response = llm.complete_json(
        system_prompt,
        {
            "task": "final_art_recipe",
            "profile": profile.model_dump(),
            "regeneration_instruction": regeneration_instruction,
            "connection_principle": "The meaning is the connection between this person and the abstract painting.",
            "thinking_mode": LLAMACPP_RECIPE_THINKING,
            "first_generation_guidance": (
                "Make this first recipe as alive and specific as a strong surprise regeneration: preserve the user's meaning, "
                "but choose a fresh central phrase, a clear visual hierarchy, and enough meaning links to make the image feel personal."
            ),
            "required_shape": {
                "title": "string",
                "central_phrase": "quote, meaningful sentence, or simple proposition",
                "main_idea": "string",
                "visual_style": "string",
                "palette": ["#hex", "#hex", "#hex"],
                "symbols": [{"visual": "specific abstract visual", "meaning": "personal connection"}],
                "composition": "string",
                "image_prompt": "string",
                "negative_prompt": "string",
                "friend_explanation": "short personal explanation",
            },
        },
        max_tokens=_recipe_token_budget(),
        response_format={"type": "json_object"},
        thinking=LLAMACPP_RECIPE_THINKING,
    )

    current_response = response
    last_error = response.error or ""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            if current_response.error:
                raise ValueError(current_response.error)
            recipe = ArtRecipe.model_validate(_normalize_recipe_payload(extract_json_object(current_response.content), profile))
            recipe = _polish_recipe(recipe)
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_attempts - 1:
                log_event(
                    "llm_art_recipe",
                    {
                        "source": current_response.source or response.source,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "error": last_error,
                        "transport_error": current_response.error,
                        "raw_content": current_response.content or response.content,
                        "retry_count": attempt,
                        "thinking_enabled": LLAMACPP_RECIPE_THINKING,
                    },
                )
                raise RuntimeError(f"Art recipe generation failed after retries: {last_error}") from exc
            current_response = llm.complete_json(
                _repair_recipe_prompt(),
                {
                    "invalid_json": current_response.content,
                    "previous_error": last_error,
                    "attempt": attempt + 1,
                    "required_shape": {
                        "title": "string",
                        "central_phrase": "string",
                        "main_idea": "string",
                        "visual_style": "string",
                        "palette": ["#hex", "#hex", "#hex"],
                        "symbols": [{"visual": "string", "meaning": "string"}],
                        "composition": "string",
                        "image_prompt": "string",
                        "negative_prompt": "string",
                        "friend_explanation": "string",
                    },
                },
                max_tokens=_recipe_token_budget(),
                response_format={"type": "json_object"},
                thinking=LLAMACPP_RECIPE_THINKING,
            )
            continue

        log_event(
            "llm_art_recipe",
            {
                "source": current_response.source,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": last_error if attempt else None,
                "transport_error": current_response.error,
                "raw_content": current_response.content,
                "retry_count": attempt,
                "thinking_enabled": LLAMACPP_RECIPE_THINKING,
                "recipe": recipe.model_dump(),
            },
        )
        return recipe
    raise RuntimeError(f"Art recipe generation failed after retries: {last_error}")


def _repair_recipe_prompt() -> str:
    return (
        "Return valid JSON only for an ArtRecipe. "
        "Required keys: title, central_phrase, main_idea, visual_style, palette, symbols, composition, "
        "image_prompt, negative_prompt, friend_explanation. "
        "symbols must be objects with visual and meaning. "
        "Make the explanation plain, personal, and connected to the user's theme."
    )


def _recipe_token_budget() -> int:
    floor = 1200 if LLAMACPP_RECIPE_THINKING else 800
    return max(LLM_RECIPE_MAX_TOKENS, floor)


def _normalize_recipe_payload(payload: dict, profile: InterviewProfile) -> dict:
    if not isinstance(payload, dict):
        return payload
    payload["title"] = str(payload.get("title", "")).strip()
    payload["central_phrase"] = str(payload.get("central_phrase", "")).strip()
    payload["main_idea"] = str(payload.get("main_idea", "")).strip()
    payload["visual_style"] = str(payload.get("visual_style", "")).strip()
    payload["composition"] = str(payload.get("composition", "")).strip()
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        symbols = []
    normalized_symbols: list[dict] = []
    for symbol in symbols:
        if isinstance(symbol, dict):
            visual = str(symbol.get("visual", "")).strip()
            meaning = str(symbol.get("meaning", "")).strip()
            if visual and meaning:
                normalized_symbols.append({"visual": visual, "meaning": meaning})
    payload["symbols"] = normalized_symbols[:12]
    payload["image_prompt"] = str(payload.get("image_prompt", "")).strip()
    payload["negative_prompt"] = str(payload.get("negative_prompt", "")).strip()
    payload["friend_explanation"] = str(payload.get("friend_explanation", "")).strip()
    return payload


def _polish_recipe(recipe: ArtRecipe) -> ArtRecipe:
    if "no text" not in recipe.image_prompt.lower():
        recipe.image_prompt = f"{recipe.image_prompt}, no text, no letters, no signature, no watermark"
    recipe.image_prompt = _avoid_literal_scene_language(recipe.image_prompt)
    recipe.image_prompt = _with_layout_diversity(recipe.image_prompt)
    recipe.negative_prompt = _with_negative_layouts(recipe.negative_prompt)
    return recipe


def _avoid_literal_scene_language(prompt: str) -> str:
    replacements = {
        "wide open field": "wide open color plane",
        "open field": "open color plane",
        "empty field": "empty color plane",
        "pale field": "pale color plane",
        "cleared field": "cleared color plane",
        "landscape format": "horizontal canvas",
        "horizon": "horizontal pressure band",
    }
    cleaned = prompt
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new).replace(old.title(), new)
    guidance = (
        "non-representational abstraction, no scenery, no literal landscape, no sky, no ground plane, "
        "no horizon line, no recognizable objects, no figures"
    )
    lowered = cleaned.lower()
    if "non-representational" in lowered and "no literal landscape" in lowered:
        return cleaned
    return f"{cleaned}, {guidance}"


def _with_layout_diversity(prompt: str) -> str:
    guidance = (
        "asymmetric composition, no four-panel grid, no equal quadrant layout, "
        "avoid centered tiled panels, one clear non-grid spatial strategy"
    )
    lowered = prompt.lower()
    if "four-panel" in lowered or "quadrant" in lowered or "non-grid" in lowered:
        return prompt
    return f"{prompt}, {guidance}"


def _with_negative_layouts(negative_prompt: str) -> str:
    additions = [
        "four equal panels",
        "four-quadrant grid",
        "split-screen",
        "tiled layout",
        "collage grid",
        "comic panels",
        "centered block grid",
        "literal landscape",
        "scenery",
        "sky",
        "ground plane",
        "horizon line",
        "mountains",
        "trees",
        "flowers",
        "buildings",
        "recognizable objects",
    ]
    lowered = negative_prompt.lower()
    missing = [item for item in additions if item not in lowered]
    if not missing:
        return negative_prompt
    return f"{negative_prompt}, {', '.join(missing)}"
