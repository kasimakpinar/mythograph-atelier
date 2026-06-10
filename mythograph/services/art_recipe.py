import time

from mythograph.schemas.art_recipe import ArtRecipe, Symbol
from mythograph.schemas.profile import InterviewProfile
from mythograph.config import LLAMACPP_RECIPE_ENABLED, LLAMACPP_RECIPE_THINKING, LLM_RECIPE_MAX_TOKENS, ROOT_DIR
from mythograph.models.llm_client import LLMClient, extract_json_object
from mythograph.services.trace_logger import log_event


PALETTES = {
    "quiet": ["#f3efe4", "#202225", "#8a8f7a", "#c7a24a", "#6d778c"],
    "bold": ["#101216", "#f04f3e", "#f6b73c", "#2f70af", "#f3efe4"],
    "mysterious": ["#0d1117", "#273043", "#7a5c8d", "#d6b35a", "#e8e1d3"],
    "clean": ["#f8f7f2", "#15191f", "#b9c2c9", "#3f6f70", "#d58b63"],
    "wild": ["#17120f", "#fb4d3d", "#1fdd9b", "#f6e05e", "#7b61ff"],
}

SYMBOL_MEANINGS = {
    "a line": "inner direction",
    "a door": "a threshold into change",
    "a flame": "chosen energy",
    "a mountain": "patient ambition",
    "a mirror": "honest self-recognition",
    "a storm": "pressure that clarifies the shape of courage",
}


def _first(items: list[str], default: str) -> str:
    return items[0] if items else default


def _style_key(profile: InterviewProfile) -> str:
    style_text = " ".join(profile.styles).lower()
    if "bold" in style_text or "dramatic" in style_text:
        return "bold"
    if "mysterious" in style_text or "symbolic" in style_text or "surprise" in style_text:
        return "mysterious"
    if "clean" in style_text or "modern" in style_text:
        return "clean"
    if "chaotic" in style_text or "wild" in style_text:
        return "wild"
    return "quiet"


def build_art_recipe(profile: InterviewProfile, regeneration_instruction: str | None = None) -> ArtRecipe:
    idea = _first(profile.ideas + profile.free_notes, "A quiet path can be more powerful than a loud success.")
    style = _first(profile.styles, "quiet and elegant")
    symbol_names = (profile.symbols or ["a line", "a door", "a flame"])[:3]

    while len(symbol_names) < 3:
        fallback = ["a line", "a door", "a flame", "a mountain", "a mirror"][len(symbol_names)]
        symbol_names.append(fallback)

    if regeneration_instruction:
        style = _adjust_style(style, regeneration_instruction)

    palette = PALETTES[_style_key(profile)]
    if regeneration_instruction and "colorful" in regeneration_instruction.lower():
        palette = PALETTES["wild"]

    symbols = [
        Symbol(visual=name, meaning=SYMBOL_MEANINGS.get(name, "a private symbol chosen by the viewer"))
        for name in symbol_names
    ]
    symbols.append(Symbol(visual="open negative space", meaning="freedom that still exists around the pressure"))

    title = _make_title(idea, symbol_names[0], regeneration_instruction)
    visual_style = _visual_style_sentence(profile, style, regeneration_instruction)
    composition = _composition_sentence(profile, symbol_names)
    image_prompt = (
        f"Single-canvas non-representational abstract contemporary painting, horizontal canvas, {visual_style}, palette {', '.join(palette)}, "
        f"{composition}. Visual symbols: {', '.join(symbol.visual for symbol in symbols)}. "
        f"Main idea: {idea}. Expressive texture, intentional negative space, non-grid spatial tension, "
        f"no text, no letters, no words, no signature, no watermark."
    )
    explanation = _friend_explanation(idea, symbols, regeneration_instruction)

    return ArtRecipe(
        title=title,
        main_idea=idea,
        visual_style=visual_style,
        palette=palette,
        symbols=symbols[:5],
        composition=composition,
        image_prompt=image_prompt,
        negative_prompt="text, letters, words, signature, watermark, literal portrait, photorealistic scene",
        friend_explanation=explanation,
    )


def build_art_recipe_with_model(
    profile: InterviewProfile,
    regeneration_instruction: str | None = None,
    client: LLMClient | None = None,
) -> ArtRecipe:
    fallback = build_art_recipe(profile, regeneration_instruction)
    llm = client or LLMClient()
    if getattr(llm, "mode", "") == "llamacpp" and not LLAMACPP_RECIPE_ENABLED:
        log_event(
            "llm_art_recipe",
            {
                "source": "deterministic",
                "elapsed_seconds": 0,
                "used_fallback": True,
                "skip_reason": "llama.cpp recipe disabled for fast MVP generation",
                "recipe": fallback.model_dump(),
            },
        )
        return fallback

    system_prompt = (ROOT_DIR / "mythograph" / "prompts" / "art_recipe_system.txt").read_text(encoding="utf-8")
    started = time.perf_counter()
    response = llm.complete_json(
        system_prompt,
        {
            "profile": profile.model_dump(),
            "regeneration_instruction": regeneration_instruction,
            "connection_principle": "The meaning is the connection between this person and the abstract painting.",
            "required_shape": {
                "title": "string",
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
    elapsed_seconds = round(time.perf_counter() - started, 3)

    if response.source == "mock":
        personalized_fallback = _personalize_recipe(fallback, profile, fallback)
        log_event(
            "llm_art_recipe",
            {
                "source": "mock",
                "elapsed_seconds": elapsed_seconds,
                "used_fallback": True,
                "thinking_enabled": LLAMACPP_RECIPE_THINKING,
                "recipe": personalized_fallback.model_dump(),
            },
        )
        return personalized_fallback

    if response.error:
        personalized_fallback = _personalize_recipe(fallback, profile, fallback)
        log_event(
            "llm_art_recipe",
            {
                "source": response.source,
                "elapsed_seconds": elapsed_seconds,
                "error": response.error,
                "transport_error": response.error,
                "raw_content": response.content,
                "used_fallback": True,
                "retry_count": 0,
                "thinking_enabled": LLAMACPP_RECIPE_THINKING,
                "recipe": personalized_fallback.model_dump(),
            },
        )
        return personalized_fallback

    try:
        recipe = ArtRecipe.model_validate(_normalize_recipe_payload(extract_json_object(response.content), profile))
        recipe = _personalize_recipe(recipe, profile, fallback)
    except Exception as exc:
        repair_response = llm.complete_json(
            _repair_recipe_prompt(),
            {
                "invalid_json": response.content,
                "fallback_recipe": fallback.model_dump(),
            },
            max_tokens=_recipe_token_budget(),
            response_format={"type": "json_object"},
            thinking=LLAMACPP_RECIPE_THINKING,
        )
        try:
            recipe = ArtRecipe.model_validate(_normalize_recipe_payload(extract_json_object(repair_response.content), profile))
            recipe = _personalize_recipe(recipe, profile, fallback)
        except Exception as repair_exc:
            error_text = f"{exc}; repair failed: {repair_exc}"
            if repair_response.error:
                error_text += f"; repair transport: {repair_response.error}"
            log_event(
                "llm_art_recipe",
                {
                    "source": repair_response.source or response.source,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "error": error_text,
                    "transport_error": repair_response.error,
                    "raw_content": repair_response.content or response.content,
                    "used_fallback": True,
                    "retry_count": 1,
                    "thinking_enabled": LLAMACPP_RECIPE_THINKING,
                    "recipe": _personalize_recipe(fallback, profile, fallback).model_dump(),
                },
            )
            return _personalize_recipe(fallback, profile, fallback)

        log_event(
            "llm_art_recipe",
            {
                "source": repair_response.source,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": str(exc),
                "transport_error": repair_response.error,
                "raw_content": repair_response.content,
                "used_fallback": False,
                "retry_count": 1,
                "thinking_enabled": LLAMACPP_RECIPE_THINKING,
                "recipe": recipe.model_dump(),
            },
        )
        return recipe

    log_event(
        "llm_art_recipe",
        {
            "source": response.source,
            "elapsed_seconds": elapsed_seconds,
            "transport_error": response.error,
            "raw_content": response.content,
            "used_fallback": False,
            "retry_count": 0,
            "thinking_enabled": LLAMACPP_RECIPE_THINKING,
            "recipe": recipe.model_dump(),
        },
    )
    return recipe


def _repair_recipe_prompt() -> str:
    return (
        "Return valid JSON only for an ArtRecipe. "
        "Required keys: title, main_idea, visual_style, palette, symbols, composition, "
        "image_prompt, negative_prompt, friend_explanation. "
        "symbols must be objects with visual and meaning. "
        "Make the explanation plain, personal, and connected to the user's theme."
    )


def _recipe_token_budget() -> int:
    floor = 560 if LLAMACPP_RECIPE_THINKING else 380
    return max(LLM_RECIPE_MAX_TOKENS, floor)


def _normalize_recipe_payload(payload: dict, profile: InterviewProfile) -> dict:
    if not isinstance(payload, dict):
        return payload
    fallback = build_art_recipe(profile)
    payload["title"] = str(payload.get("title", "")).strip() or fallback.title
    payload["main_idea"] = str(payload.get("main_idea", "")).strip() or fallback.main_idea
    payload["visual_style"] = str(payload.get("visual_style", "")).strip() or fallback.visual_style
    payload["composition"] = str(payload.get("composition", "")).strip() or fallback.composition
    if not isinstance(payload.get("palette"), list) or not payload["palette"]:
        payload["palette"] = fallback.palette
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
    for symbol in _symbols_from_profile(profile):
        if len(normalized_symbols) >= 3:
            break
        if all(existing["visual"].lower() != symbol.visual.lower() for existing in normalized_symbols):
            normalized_symbols.append(symbol.model_dump())
    payload["symbols"] = normalized_symbols[:6]
    payload["image_prompt"] = str(payload.get("image_prompt", "")).strip() or _image_prompt_from_partial(payload, fallback)
    if not payload.get("negative_prompt"):
        payload["negative_prompt"] = "text, letters, words, signature, watermark, literal portrait, photorealistic scene"
    if not payload.get("friend_explanation"):
        payload["friend_explanation"] = _personal_connection_explanation(profile, ArtRecipe.model_validate(fallback.model_dump()))
    return payload


def _image_prompt_from_partial(payload: dict, fallback: ArtRecipe) -> str:
    style = str(payload.get("visual_style", "")).strip() or fallback.visual_style
    composition = str(payload.get("composition", "")).strip() or fallback.composition
    idea = str(payload.get("main_idea", "")).strip() or fallback.main_idea
    visual_symbols = []
    symbols = payload.get("symbols")
    if isinstance(symbols, list):
        for symbol in symbols:
            if isinstance(symbol, dict) and str(symbol.get("visual", "")).strip():
                visual_symbols.append(str(symbol["visual"]).strip())
    if not visual_symbols:
        visual_symbols = [symbol.visual for symbol in fallback.symbols[:3]]
    return (
        f"Single-canvas non-representational abstract contemporary painting, horizontal canvas, {style}, {composition}. "
        f"Visual symbols: {', '.join(visual_symbols)}. Main idea: {idea}. "
        "No grid, no panels, no text, expressive texture, intentional negative space."
    )


def _personalize_recipe(recipe: ArtRecipe, profile: InterviewProfile, fallback: ArtRecipe) -> ArtRecipe:
    recipe.title = recipe.title.strip() or fallback.title
    recipe.main_idea = recipe.main_idea.strip() or _first(profile.ideas + profile.free_notes, fallback.main_idea)
    recipe.visual_style = recipe.visual_style.strip() or fallback.visual_style
    recipe.composition = recipe.composition.strip() or fallback.composition
    recipe.image_prompt = recipe.image_prompt.strip() or fallback.image_prompt
    recipe.negative_prompt = recipe.negative_prompt.strip() or fallback.negative_prompt

    if (
        _uses_stock_explanation(recipe.friend_explanation)
        or _sounds_robotic(recipe.friend_explanation)
        or _explanation_too_thin(recipe.friend_explanation)
    ):
        recipe.friend_explanation = _personal_connection_explanation(profile, recipe)

    chosen_symbol_text = " ".join(profile.symbols).lower()
    if not chosen_symbol_text and _uses_stock_symbol_set(recipe):
        recipe.symbols = _symbols_from_profile(profile)

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


def _uses_stock_explanation(text: str) -> bool:
    lowered = (text or "").lower()
    return (
        "is not decoration" in lowered
        or "a door gives it a second force" in lowered
        or "threshold into change" in lowered
    )


def _sounds_robotic(text: str) -> bool:
    lowered = (text or "").lower()
    return (
        "that is the idea behind the painting" in lowered
        or "this painting turns" in lowered
        or "the meaning is not hidden" in lowered
    )


def _explanation_too_thin(text: str) -> bool:
    words = (text or "").split()
    return len(words) < 24


def _uses_stock_symbol_set(recipe: ArtRecipe) -> bool:
    visuals = {symbol.visual.lower().strip() for symbol in recipe.symbols[:3]}
    return {"a line", "a door", "a flame"}.issubset(visuals)


def _symbols_from_profile(profile: InterviewProfile) -> list[Symbol]:
    idea_text = " ".join(profile.ideas + profile.free_notes).lower()
    palette = str(profile.visual_preferences.get("palette_mood", "")).strip()
    if any(word in idea_text for word in ["weather", "light", "bright", "joy", "relief", "fresh", "clean"]):
        return [
            Symbol(visual="clean reflected light", meaning="joy arriving as a change in atmosphere"),
            Symbol(visual="small bright ripples", meaning="happiness spreading without needing to shout"),
            Symbol(visual=f"{palette or 'hushed'} color weather", meaning="the private climate of relief"),
        ]
    if "lonely" in idea_text or "loneliness" in idea_text or "silence" in idea_text:
        return [
            Symbol(visual="a pale open color plane", meaning="room around the feeling instead of escape from it"),
            Symbol(visual="a small echoing mark", meaning="the self answering back softly"),
            Symbol(visual=f"{palette or 'soft'} color hush", meaning="peace arriving as atmosphere, not explanation"),
        ]
    if "positive" in idea_text or "hope" in idea_text:
        return [
            Symbol(visual="a rising warm mark", meaning="hope becoming visible before it becomes loud"),
            Symbol(visual="a clear gap of light", meaning="space for the next good thing"),
            Symbol(visual=f"{palette or 'bright'} pulse", meaning="optimism held in color"),
        ]
    return [
        Symbol(visual="an open color plane", meaning="the viewer's private room inside the painting"),
        Symbol(visual="one recurring mark", meaning="the thought that keeps returning"),
        Symbol(visual=f"{palette or 'chosen'} color pressure", meaning="emotion translated into atmosphere"),
    ]


def _personal_connection_explanation(profile: InterviewProfile, recipe: ArtRecipe) -> str:
    idea = _first(profile.ideas + profile.free_notes, recipe.main_idea)
    chosen_items = [item for item in profile.ideas[1:] + profile.free_notes[-3:] if item != "visual preferences"]
    chosen = ", ".join(chosen_items)
    palette = str(profile.visual_preferences.get("palette_mood", "")).strip()
    idea_text = idea.lower()
    if any(word in idea_text for word in ["weather", "light", "bright", "joy", "relief", "fresh", "clean"]):
        detail = chosen or palette or "the softened rhythm"
        return (
            f"The atmosphere here feels like permission to brighten. "
            f"{detail.capitalize()} becomes the point where the painting meets the viewer: a quiet joy that spreads outward, "
            f"as if the feeling has found cleaner air."
        )
    if "lonely" in idea_text or "loneliness" in idea_text:
        detail = chosen or palette or "the open space"
        return (
            f"This piece does not try to cure loneliness; it gives it a gentle room. "
            f"{detail.capitalize()} becomes the place where peace can sit beside solitude without needing to explain itself."
        )
    if chosen:
        return (
            f"The painting gives {_clean_idea_for_explanation(idea)} a body: {chosen.lower()} become rhythm, distance, and color. "
            f"It works when the viewer feels those abstract marks answer something they already carried in quietly."
        )
    if palette:
        return (
            f"The {palette} atmosphere lets {_clean_idea_for_explanation(idea)} stay quiet without becoming vague. "
            f"The connection happens when the viewer recognizes the mood before they can name it."
        )
    return (
        f"The painting gives {_clean_idea_for_explanation(idea)} a place to live outside language. "
        f"Its meaning arrives when the abstract space feels strangely familiar to the person looking at it."
    )


def _clean_idea_for_explanation(idea: str) -> str:
    clean = " ".join((idea or "").strip().split())
    lowered = clean.lower()
    for prefix in ["i want something about ", "i want something ", "something about ", "i want "]:
        if lowered.startswith(prefix):
            return clean[len(prefix) :].strip().lower() or clean.lower()
    return clean.lower()


def _adjust_style(style: str, instruction: str) -> str:
    lowered = instruction.lower()
    if "elegant" in lowered:
        return "quiet and elegant"
    if "intense" in lowered:
        return "bold and dramatic"
    if "minimal" in lowered:
        return "clean and modern"
    if "mysterious" in lowered:
        return "mysterious and symbolic"
    if "colorful" in lowered:
        return "chaotic but beautiful"
    if "surprise" in lowered:
        return "strange, vivid, and symbolic"
    return style


def _make_title(idea: str, symbol: str, regeneration_instruction: str | None) -> str:
    if "control" in idea.lower() or "response" in idea.lower():
        return "The Line That Did Not Bend"
    if "meaning" in idea.lower():
        return "A Map Drawn Inside the Noise"
    if "quiet" in idea.lower():
        return "The Quiet Path"
    if regeneration_instruction and "style" in regeneration_instruction.lower():
        return "The Same Myth in New Weather"
    return f"The Shape of {symbol.replace('a ', '').title()}"


def _visual_style_sentence(profile: InterviewProfile, style: str, regeneration_instruction: str | None) -> str:
    prefs = profile.visual_preferences
    complexity = "minimal" if prefs.get("minimal_rich", 35) < 50 else "layered"
    energy = "calm" if prefs.get("calm_intense", 45) < 55 else "intense"
    geometry = "geometric" if prefs.get("geometric_organic", 45) < 55 else "organic"
    base = f"{complexity}, {energy}, {geometry}, {style}"
    if regeneration_instruction:
        return f"{base}, revised to {regeneration_instruction.lower()}"
    return base


def _composition_sentence(profile: InterviewProfile, symbols: list[str]) -> str:
    strategies = [
        f"an off-center color plane pulled toward the lower right by {symbols[0]}, with a long quiet margin above it",
        f"a diagonal drift of interrupted marks around {symbols[0]}, leaving one open path through the canvas",
        f"a low horizontal pressure band with {symbols[0]} suspended above it and sparse echoes near the edges",
        f"a broken ring of texture orbiting {symbols[0]}, with the center deliberately left unresolved",
        f"a cascade of thin color planes descending from one edge, interrupted by {symbols[0]} before it reaches the center",
        f"one dense corner pushing against a wide empty color plane, with {symbols[0]} acting as the hinge",
    ]
    signature = "|".join(profile.ideas + profile.symbols + profile.free_notes + [str(profile.visual_preferences)])
    index = sum(ord(char) for char in signature) % len(strategies)
    return strategies[index]


def _friend_explanation(idea: str, symbols: list[Symbol], regeneration_instruction: str | None) -> str:
    first, second = symbols[0], symbols[1]
    sentence = (
        f"{_as_subject(first.visual)} is not decoration; it is {first.meaning}. "
        f"{_as_subject(second.visual)} gives it a second force: {second.meaning}. "
        f"That is the idea behind the painting: {idea.lower()}"
    )
    if regeneration_instruction and "explanation" in regeneration_instruction.lower():
        sentence += " It is meant to feel like a private myth made visible for a moment."
    return sentence


def _as_subject(visual: str) -> str:
    return visual[:1].upper() + visual[1:] if visual.startswith(("a ", "an ")) else f"The {visual}"
