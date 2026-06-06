from mythograph.schemas.art_recipe import ArtRecipe, Symbol
from mythograph.schemas.profile import InterviewProfile


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
        f"Abstract contemporary painting, {visual_style}, palette {', '.join(palette)}, "
        f"{composition}. Visual symbols: {', '.join(symbol.visual for symbol in symbols)}. "
        f"Main idea: {idea}. Museum-quality abstract wall art, balanced negative space, expressive texture, "
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
    prefs = profile.visual_preferences
    if prefs.get("geometric_organic", 45) < 55:
        return f"asymmetric blocks crossed by {symbols[0]} with a quiet field of empty space"
    return f"flowing layered forms orbiting {symbols[0]} with textured edges and one clear focal opening"


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
