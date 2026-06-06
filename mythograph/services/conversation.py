import time

from mythograph.config import CONVERSATION_MODE, ROOT_DIR
from mythograph.models.llm_client import LLMClient, extract_json_object
from mythograph.schemas.profile import InterviewProfile
from mythograph.schemas.ui import ControlKind, ControlResponse, ConversationTurn, DynamicControl, SliderSpec
from mythograph.services.interview import (
    CONTRAST_OPTIONS,
    IDEA_OPTIONS,
    STYLE_OPTIONS,
    SYMBOL_OPTIONS,
    new_profile,
    update_scores,
)
from mythograph.services.trace_logger import log_event


PALETTE_OPTIONS = [
    "bone, ink, muted gold",
    "deep blue, violet, candle gold",
    "warm white, clay, black",
    "charcoal, mineral green, pale sky",
    "surprise me with tension",
]


def start_session() -> ConversationTurn:
    return ConversationTurn(
        assistant_message="Tell me what this painting should hold. One sentence is enough.",
        progress_label="Meaning: waiting for the first signal",
        reason="new session",
        controls=[
            DynamicControl(
                kind=ControlKind.TEXT_REFINEMENT,
                label="Starting thought",
                prompt="What should the painting be about?",
            )
        ],
    )


def advance_conversation(
    profile: InterviewProfile | None = None,
    user_message: str = "",
    control_response: ControlResponse | None = None,
    chat_history: list[dict[str, str]] | None = None,
    client: LLMClient | None = None,
) -> tuple[InterviewProfile, ConversationTurn]:
    profile = profile or new_profile()
    clean_message = (user_message or "").strip()

    if clean_message:
        _apply_free_text(profile, clean_message)

    if control_response:
        apply_control_response(profile, control_response)

    profile = update_scores(profile)
    fallback_turn = choose_conversation_turn(profile)
    turn = choose_conversation_turn_with_model(profile, fallback_turn, chat_history or [], client)
    log_event(
        "conversation_turn",
        {
            "profile": profile.model_dump(),
            "turn": turn.model_dump(),
        },
    )
    log_event(
        "generation_readiness",
        {
            "ready": profile.scores.ready_to_generate,
            "scores": profile.scores.model_dump(),
            "turn_count": profile.turn_count,
        },
    )
    return profile, turn


def choose_conversation_turn_with_model(
    profile: InterviewProfile,
    fallback_turn: ConversationTurn,
    chat_history: list[dict[str, str]] | None = None,
    client: LLMClient | None = None,
) -> ConversationTurn:
    if CONVERSATION_MODE != "model_assisted":
        return fallback_turn

    llm = client or LLMClient()
    system_prompt = (ROOT_DIR / "mythograph" / "prompts" / "conversation_director_system.txt").read_text(
        encoding="utf-8"
    )
    started = time.perf_counter()
    response = llm.complete_json(
        system_prompt,
        {
            "profile": profile.model_dump(),
            "chat_history": (chat_history or [])[-10:],
            "fallback_turn": fallback_turn.model_dump(),
            "allowed_control_kinds": [kind.value for kind in ControlKind],
            "available_option_sets": {
                "ideas": IDEA_OPTIONS,
                "styles": STYLE_OPTIONS,
                "symbols": SYMBOL_OPTIONS,
                "contrasts": CONTRAST_OPTIONS,
                "palette_moods": PALETTE_OPTIONS,
            },
        },
    )
    elapsed_seconds = round(time.perf_counter() - started, 3)

    if response.source == "mock" or response.error:
        log_event(
            "llm_conversation_turn",
            {
                "source": response.source,
                "elapsed_seconds": elapsed_seconds,
                "error": response.error,
                "transport_error": response.error,
                "raw_content": response.content,
                "used_fallback": True,
                "turn": fallback_turn.model_dump(),
            },
        )
        return fallback_turn

    try:
        candidate = ConversationTurn.model_validate(extract_json_object(response.content))
        candidate = sanitize_conversation_turn(candidate, fallback_turn)
    except Exception as exc:
        log_event(
            "llm_conversation_turn",
            {
                "source": response.source,
                "elapsed_seconds": elapsed_seconds,
                "error": str(exc),
                "transport_error": response.error,
                "raw_content": response.content,
                "used_fallback": True,
                "turn": fallback_turn.model_dump(),
            },
        )
        return fallback_turn

    log_event(
        "llm_conversation_turn",
        {
            "source": response.source,
            "elapsed_seconds": elapsed_seconds,
            "transport_error": response.error,
            "raw_content": response.content,
            "used_fallback": False,
            "turn": candidate.model_dump(),
        },
    )
    return candidate


def sanitize_conversation_turn(candidate: ConversationTurn, fallback_turn: ConversationTurn) -> ConversationTurn:
    if not candidate.controls:
        candidate.controls = fallback_turn.controls
    candidate.controls = [candidate.controls[0]]
    control = candidate.controls[0]

    if candidate.is_ready and not fallback_turn.is_ready:
        return fallback_turn
    if fallback_turn.is_ready:
        candidate.is_ready = True
        candidate.controls = [
            DynamicControl(
                kind=ControlKind.READY_BUTTON,
                label="Create artwork",
                prompt="Generate the painting",
                options=["Create artwork"],
            )
        ]
        return candidate

    if control.kind == ControlKind.READY_BUTTON:
        return fallback_turn

    if control.kind in {ControlKind.CHOICE_CARDS, ControlKind.MULTI_CHOICE_CARDS, ControlKind.SWATCH_PICKER}:
        control.options = _sanitize_options(control.options, fallback_turn.controls[0].options if fallback_turn.controls else [])
    elif control.kind == ControlKind.SLIDER_GROUP:
        control.sliders = _sanitize_sliders(control.sliders)
    elif control.kind == ControlKind.TEXT_REFINEMENT:
        control.options = []
        control.sliders = []

    if control.kind in {ControlKind.CHOICE_CARDS, ControlKind.MULTI_CHOICE_CARDS, ControlKind.SWATCH_PICKER} and not control.options:
        return fallback_turn
    if control.kind == ControlKind.SLIDER_GROUP and not control.sliders:
        return fallback_turn
    return candidate


def _sanitize_options(options: list[str], fallback_options: list[str]) -> list[str]:
    cleaned: list[str] = []
    for option in options:
        value = str(option).strip()
        if value and value not in cleaned:
            cleaned.append(value)
        if len(cleaned) >= 6:
            break
    if cleaned:
        return cleaned
    return fallback_options[:6]


def _sanitize_sliders(sliders: list[SliderSpec]) -> list[SliderSpec]:
    cleaned: list[SliderSpec] = []
    for slider in sliders:
        key = slider.key.strip() or f"slider_{len(cleaned) + 1}"
        label = slider.label.strip() or key.replace("_", " ").title()
        left_label = slider.left_label.strip() or "less"
        right_label = slider.right_label.strip() or "more"
        value = max(0, min(100, int(slider.value)))
        cleaned.append(
            SliderSpec(
                key=key,
                label=label,
                left_label=left_label,
                right_label=right_label,
                value=value,
            )
        )
        if len(cleaned) >= 3:
            break
    return cleaned


def should_generate(profile: InterviewProfile) -> bool:
    return update_scores(profile).scores.ready_to_generate


def apply_control_response(profile: InterviewProfile, response: ControlResponse) -> InterviewProfile:
    values = [value.strip() for value in response.values if value and value.strip()]
    text = response.text.strip()
    turn_added = False

    if response.kind == ControlKind.MULTI_CHOICE_CARDS:
        if all(value in SYMBOL_OPTIONS for value in values):
            profile.symbols.extend(values)
        else:
            profile.ideas.extend(values)
    elif response.kind == ControlKind.CHOICE_CARDS:
        _apply_choice(profile, values[0] if values else "")
    elif response.kind == ControlKind.SLIDER_GROUP:
        profile.visual_preferences.update(response.sliders)
        if "visual preferences" not in profile.free_notes:
            profile.free_notes.append("visual preferences")
    elif response.kind == ControlKind.SWATCH_PICKER:
        if values:
            profile.visual_preferences["palette_mood"] = values[0]
    elif response.kind == ControlKind.TEXT_REFINEMENT and text:
        _apply_free_text(profile, text)
        turn_added = True

    if not turn_added:
        profile.turn_count += 1
    updated = update_scores(profile)
    log_event("control_response", {"response": response.model_dump(), "profile": updated.model_dump()})
    log_event(
        "generation_readiness",
        {
            "ready": updated.scores.ready_to_generate,
            "scores": updated.scores.model_dump(),
            "turn_count": updated.turn_count,
        },
    )
    return updated


def choose_conversation_turn(profile: InterviewProfile) -> ConversationTurn:
    profile = update_scores(profile)

    if profile.scores.ready_to_generate:
        return ConversationTurn(
            assistant_message="I have enough: a meaning, a visual taste, and symbols. I can paint it now.",
            progress_label="Ready: meaning, taste, and symbols are locked",
            reason="profile has enough signal for generation",
            is_ready=True,
            controls=[
                DynamicControl(
                    kind=ControlKind.READY_BUTTON,
                    label="Create artwork",
                    prompt="Generate the painting",
                    options=["Create artwork"],
                )
            ],
        )

    if profile.scores.idea_anchor < 0.65:
        return ConversationTurn(
            assistant_message="Good. Choose the idea that should sit closest to the center of the painting.",
            progress_label="Meaning: choose the core pressure",
            reason="idea anchor needs one more signal",
            controls=[
                DynamicControl(
                    kind=ControlKind.MULTI_CHOICE_CARDS,
                    label="Core meaning",
                    prompt="Pick one or two.",
                    options=IDEA_OPTIONS,
                )
            ],
        )

    if profile.scores.visual_taste < 0.45:
        return ConversationTurn(
            assistant_message="Now shape the visual temperament. I only need a quick taste profile.",
            progress_label="Taste: tune the visual language",
            reason="visual taste is not specific enough",
            controls=[
                DynamicControl(
                    kind=ControlKind.SLIDER_GROUP,
                    label="Visual temperament",
                    prompt="Move the three studio dials.",
                    sliders=[
                        SliderSpec(key="minimal_rich", label="Density", left_label="minimal", right_label="rich", value=35),
                        SliderSpec(key="calm_intense", label="Energy", left_label="calm", right_label="intense", value=45),
                        SliderSpec(key="geometric_organic", label="Shape", left_label="geometric", right_label="organic", value=45),
                    ],
                )
            ],
        )

    if "palette_mood" not in profile.visual_preferences:
        return ConversationTurn(
            assistant_message="Pick a color weather for the piece. This does not lock exact colors; it gives the model a mood.",
            progress_label="Taste: choose color weather",
            reason="palette mood is missing",
            controls=[
                DynamicControl(
                    kind=ControlKind.SWATCH_PICKER,
                    label="Color weather",
                    prompt="Choose one palette mood.",
                    options=PALETTE_OPTIONS,
                )
            ],
        )

    if not profile.styles:
        return ConversationTurn(
            assistant_message="Which presence should the artwork have when someone first sees it?",
            progress_label="Taste: choose the artwork presence",
            reason="style language is missing",
            controls=[
                DynamicControl(
                    kind=ControlKind.CHOICE_CARDS,
                    label="Presence",
                    prompt="Pick one.",
                    options=STYLE_OPTIONS,
                )
            ],
        )

    if profile.scores.symbolic_material < 0.55:
        return ConversationTurn(
            assistant_message="Give me one symbol to transform. It can stay abstract, but it needs a visual anchor.",
            progress_label="Symbol: choose the anchor",
            reason="symbolic material needs a stronger anchor",
            controls=[
                DynamicControl(
                    kind=ControlKind.MULTI_CHOICE_CARDS,
                    label="Symbolic anchors",
                    prompt="Pick one or two.",
                    options=SYMBOL_OPTIONS,
                )
            ],
        )

    return ConversationTurn(
        assistant_message="One last pressure point: should the painting comfort you, challenge you, or surprise you?",
        progress_label="Readiness: sharpen the emotional angle",
        reason="contrast can sharpen the final recipe",
        controls=[
            DynamicControl(
                kind=ControlKind.CHOICE_CARDS,
                label="Emotional angle",
                prompt="Pick one.",
                options=CONTRAST_OPTIONS,
            )
        ],
    )


def _apply_free_text(profile: InterviewProfile, text: str) -> None:
    if not profile.ideas:
        profile.ideas.append(text)
    else:
        profile.free_notes.append(text)
    profile.turn_count += 1


def _apply_choice(profile: InterviewProfile, value: str) -> None:
    if not value:
        return
    if value in STYLE_OPTIONS:
        profile.styles.append(value)
    elif value in SYMBOL_OPTIONS:
        profile.symbols.append(value)
    elif value in CONTRAST_OPTIONS:
        profile.contrasts.append(value)
    elif value in PALETTE_OPTIONS:
        profile.visual_preferences["palette_mood"] = value
    else:
        profile.free_notes.append(value)
