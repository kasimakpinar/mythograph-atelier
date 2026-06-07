import time

from mythograph.config import CONVERSATION_MODE, LLAMACPP_CHAT_ENABLED, LLM_CHAT_MAX_TOKENS, ROOT_DIR
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
    turn = choose_conversation_turn_with_model(profile, fallback_turn, client)
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
    client: LLMClient | None = None,
) -> ConversationTurn:
    if CONVERSATION_MODE != "model_assisted":
        return fallback_turn

    llm = client or LLMClient()
    if getattr(llm, "mode", "") == "llamacpp" and not LLAMACPP_CHAT_ENABLED:
        log_event(
            "llm_conversation_turn",
            {
                "source": "deterministic",
                "elapsed_seconds": 0,
                "used_fallback": True,
                "skip_reason": "llama.cpp chat disabled for fast interactive turns",
                "atelier_state": build_atelier_state(profile),
                "turn": fallback_turn.model_dump(),
            },
        )
        return fallback_turn

    system_prompt = (ROOT_DIR / "mythograph" / "prompts" / "conversation_director_system.txt").read_text(
        encoding="utf-8"
    )
    started = time.perf_counter()
    payload = {
        "atelier_state": build_atelier_state(profile),
        "fallback_turn": fallback_turn.model_dump(),
        "allowed_control_kinds": [kind.value for kind in ControlKind],
        "available_option_sets": {
            "ideas": IDEA_OPTIONS,
            "styles": STYLE_OPTIONS,
            "symbols": SYMBOL_OPTIONS,
            "contrasts": CONTRAST_OPTIONS,
            "palette_moods": PALETTE_OPTIONS,
        },
    }
    response = llm.complete_json(
        system_prompt,
        payload,
        max_tokens=LLM_CHAT_MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    if response.source == "mock":
        return fallback_turn

    if response.error:
        elapsed_seconds = round(time.perf_counter() - started, 3)
        error_turn = model_error_turn(
            "The text model could not answer cleanly. Try one shorter phrase.",
            response.error,
        )
        log_event(
            "llm_conversation_turn",
            {
                "source": response.source,
                "elapsed_seconds": elapsed_seconds,
                "error": response.error,
                "transport_error": response.error,
                "raw_content": response.content,
                "used_fallback": False,
                "retry_count": 0,
                "chosen_control": error_turn.controls[0].kind if error_turn.controls else None,
                "atelier_state": build_atelier_state(profile),
                "turn": error_turn.model_dump(),
            },
        )
        return error_turn

    try:
        candidate = ConversationTurn.model_validate(extract_json_object(response.content))
        candidate = sanitize_conversation_turn(candidate, fallback_turn)
    except Exception as exc:
        repair_response = llm.complete_json(
            _repair_system_prompt(),
            {
                "invalid_json": response.content,
                "required_shape": "ConversationTurn JSON with assistant_message, progress_label, reason, is_ready, controls.",
                "fallback_turn": fallback_turn.model_dump(),
            },
            max_tokens=LLM_CHAT_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        try:
            candidate = ConversationTurn.model_validate(extract_json_object(repair_response.content))
            candidate = sanitize_conversation_turn(candidate, fallback_turn)
        except Exception as repair_exc:
            elapsed_seconds = round(time.perf_counter() - started, 3)
            raw_content = repair_response.content or response.content
            error_text = f"{exc}; repair failed: {repair_exc}"
            if repair_response.error:
                error_text += f"; repair transport: {repair_response.error}"
            error_turn = model_error_turn(
                "The text model drifted outside the UI schema. Try one sharper sentence.",
                error_text,
            )
            log_event(
                "llm_conversation_turn",
                {
                    "source": repair_response.source or response.source,
                    "elapsed_seconds": elapsed_seconds,
                    "error": error_text,
                    "transport_error": repair_response.error,
                    "raw_content": raw_content,
                    "used_fallback": False,
                    "retry_count": 1,
                    "chosen_control": error_turn.controls[0].kind if error_turn.controls else None,
                    "atelier_state": build_atelier_state(profile),
                    "turn": error_turn.model_dump(),
                },
            )
            return error_turn

        elapsed_seconds = round(time.perf_counter() - started, 3)
        log_event(
            "llm_conversation_turn",
            {
                "source": repair_response.source,
                "elapsed_seconds": elapsed_seconds,
                "error": str(exc),
                "transport_error": repair_response.error,
                "raw_content": repair_response.content,
                "used_fallback": False,
                "retry_count": 1,
                "chosen_control": candidate.controls[0].kind if candidate.controls else None,
                "atelier_state": build_atelier_state(profile),
                "turn": candidate.model_dump(),
            },
        )
        return candidate

    elapsed_seconds = round(time.perf_counter() - started, 3)
    log_event(
        "llm_conversation_turn",
        {
            "source": response.source,
            "elapsed_seconds": elapsed_seconds,
            "transport_error": response.error,
            "raw_content": response.content,
            "used_fallback": False,
            "retry_count": 0,
            "chosen_control": candidate.controls[0].kind if candidate.controls else None,
            "atelier_state": build_atelier_state(profile),
            "turn": candidate.model_dump(),
        },
    )
    return candidate


def model_error_turn(message: str, detail: str = "") -> ConversationTurn:
    reason = "llama.cpp did not return valid schema JSON"
    if detail:
        reason = f"{reason}: {detail[:900]}"
    return ConversationTurn(
        assistant_message=message,
        progress_label="Model: retry needed",
        reason=reason,
        is_ready=False,
        controls=[
            DynamicControl(
                kind=ControlKind.TEXT_REFINEMENT,
                label="Retry",
                prompt="Send a shorter signal and the atelier will ask again.",
            )
        ],
    )


def _repair_system_prompt() -> str:
    return (
        "Return only valid JSON for Mythograph Atelier. No markdown. "
        "Keep one control only. Use allowed kind names exactly. "
        "If unsure, use text_refinement with empty options and sliders."
    )


def build_atelier_state(profile: InterviewProfile) -> dict:
    visual_preferences = dict(profile.visual_preferences)
    answers = profile.ideas + profile.styles + profile.symbols + profile.contrasts + profile.free_notes
    return {
        "mood": _first_nonempty(profile.contrasts + profile.styles),
        "main_idea": _first_nonempty(profile.ideas),
        "main_symbol": _first_nonempty(profile.symbols),
        "palette_mood": visual_preferences.get("palette_mood", ""),
        "visual_style": _first_nonempty(profile.styles),
        "visual_preferences": visual_preferences,
        "answers_so_far": [answer[:180] for answer in answers[-6:]],
        "scores": profile.scores.model_dump(),
        "turn_count": profile.turn_count,
    }


def _first_nonempty(items: list[str]) -> str:
    for item in items:
        if item:
            return item
    return ""


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
    idea = _first_nonempty(profile.ideas + profile.free_notes)
    idea_fragment = _short_fragment(idea)

    if profile.scores.ready_to_generate:
        return ConversationTurn(
            assistant_message=f"I have enough to paint {idea_fragment or 'this private myth'} now.",
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
            assistant_message=f"Good. What kind of order should {idea_fragment or 'this'} reveal?",
            progress_label="Meaning: choose the core pressure",
            reason="idea anchor needs one more signal",
            controls=[
                DynamicControl(
                    kind=ControlKind.MULTI_CHOICE_CARDS,
                    label="Core meaning",
                    prompt="Pick one or two.",
                    options=_idea_options_for(profile),
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
            assistant_message=f"Pick the color weather around {idea_fragment or 'the central feeling'}.",
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
            assistant_message=f"When someone first sees {idea_fragment or 'the work'}, what presence should it have?",
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
            assistant_message=f"Choose one visual anchor that can carry {idea_fragment or 'the meaning'}.",
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
        assistant_message=f"One last pressure point: how should {idea_fragment or 'the painting'} treat the viewer?",
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


def _idea_options_for(profile: InterviewProfile) -> list[str]:
    idea = _first_nonempty(profile.ideas)
    if not idea:
        return IDEA_OPTIONS
    fragment = _short_fragment(idea)
    return [
        f"{fragment} has a hidden structure.",
        f"{fragment} is a calm center inside pressure.",
        f"{fragment} becomes visible only through contrast.",
        f"{fragment} needs one clear path through noise.",
        "Let the painting find the order by surprise.",
    ]


def _short_fragment(text: str) -> str:
    clean = " ".join((text or "").strip().split())
    if not clean:
        return ""
    if len(clean) <= 44:
        return clean
    return clean[:41].rstrip(" ,.;:") + "..."
