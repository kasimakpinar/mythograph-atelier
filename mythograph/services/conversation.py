import time

from mythograph.config import CONVERSATION_MODE, LLAMACPP_CHAT_ENABLED, LLM_CHAT_MAX_TOKENS, ROOT_DIR
from mythograph.models.llm_client import LLMClient, extract_json_object
from mythograph.schemas.profile import InterviewProfile
from mythograph.schemas.ui import ControlKind, ControlResponse, ConversationTurn, DynamicControl, SliderSpec
from mythograph.services.interview import CONTRAST_OPTIONS, STYLE_OPTIONS, SYMBOL_OPTIONS, new_profile, update_scores
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
    _remember_model_turn(profile, turn)
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
        "next_need": _next_need_from_turn(fallback_turn),
        "can_generate": fallback_turn.is_ready,
        "allowed_control_kinds": [kind.value for kind in ControlKind],
        "control_guidance": {
            "choice_cards": "3-5 fresh, specific options for one choice.",
            "multi_choice_cards": "3-5 fresh, specific options; user may pick two.",
            "swatch_picker": "3-5 short palette moods, not repeated from previous turns.",
            "slider_group": "2-3 visual dials with expressive labels.",
            "text_refinement": "one short free-text prompt when user intent is vague.",
            "ready_button": "only when can_generate is true.",
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
                "raw_preview": _preview(response.content),
                "used_fallback": False,
                "retry_count": 0,
                "chosen_control": error_turn.controls[0].kind if error_turn.controls else None,
                "atelier_state": build_atelier_state(profile),
                "turn": error_turn.model_dump(),
            },
        )
        return error_turn

    try:
        candidate = ConversationTurn.model_validate(
            _normalize_turn_payload(extract_json_object(response.content), fallback_turn)
        )
        candidate = sanitize_conversation_turn(candidate, fallback_turn, profile)
    except Exception as exc:
        repair_response = llm.complete_json(
            _repair_system_prompt(),
            {
                "invalid_json": response.content,
                "required_shape": "ConversationTurn JSON with assistant_message, progress_label, reason, is_ready, controls.",
                "next_need": _next_need_from_turn(fallback_turn),
                "can_generate": fallback_turn.is_ready,
            },
            max_tokens=LLM_CHAT_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        try:
            candidate = ConversationTurn.model_validate(
                _normalize_turn_payload(extract_json_object(repair_response.content), fallback_turn)
            )
            candidate = sanitize_conversation_turn(candidate, fallback_turn, profile)
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
                    "raw_preview": _preview(raw_content),
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
                "raw_preview": _preview(repair_response.content),
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
            "raw_preview": _preview(response.content),
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


def _preview(text: str, limit: int = 900) -> str:
    return " ".join((text or "").split())[:limit]


def _repair_system_prompt() -> str:
    return (
        "Return only valid JSON for Mythograph Atelier. No markdown. "
        "Keep one control only. Use allowed kind names exactly. "
        "Use the suggested_component from next_need unless can_generate is true. "
        "Do not repeat previous user answers as options. "
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
        "already_chosen": _recent_choices(profile),
        "avoid_questions": profile.asked_questions[-5:],
        "avoid_options": profile.offered_options[-16:],
    }


def _normalize_turn_payload(payload: dict, fallback_turn: ConversationTurn) -> dict:
    if not isinstance(payload, dict):
        return payload
    controls = payload.get("controls")
    if not isinstance(controls, list) or not controls:
        return payload

    control = controls[0]
    if not isinstance(control, dict):
        return payload

    if not str(control.get("prompt", "")).strip():
        control["prompt"] = _prompt_from_control(control, payload, fallback_turn)

    kind = str(control.get("kind", ""))
    if kind == ControlKind.SLIDER_GROUP.value:
        control["options"] = []
        control["sliders"] = _normalize_slider_payload(control.get("sliders"), control)
    elif kind == ControlKind.TEXT_REFINEMENT.value:
        control["options"] = []
        control["sliders"] = []
    elif kind in {
        ControlKind.CHOICE_CARDS.value,
        ControlKind.MULTI_CHOICE_CARDS.value,
        ControlKind.SWATCH_PICKER.value,
        ControlKind.READY_BUTTON.value,
    }:
        control["sliders"] = []
        if not isinstance(control.get("options"), list):
            control["options"] = []
    controls[0] = control
    payload["controls"] = [control]
    return payload


def _prompt_from_control(control: dict, payload: dict, fallback_turn: ConversationTurn) -> str:
    for key in ("prompt", "label"):
        value = str(control.get(key, "")).strip()
        if value:
            return value
    message = str(payload.get("assistant_message", "")).strip()
    if message:
        return message
    if fallback_turn.controls:
        return fallback_turn.controls[0].prompt
    return "Choose one direction."


def _normalize_slider_payload(sliders: object, control: dict) -> list[dict]:
    if isinstance(sliders, list) and sliders and all(isinstance(slider, dict) for slider in sliders):
        return sliders[:3]

    if isinstance(sliders, list) and sliders and all(isinstance(slider, str) for slider in sliders):
        return [_slider_from_name(name, index) for index, name in enumerate(sliders[:3])]

    label = str(control.get("label", "")).strip() or "Intensity"
    left_label = str(control.get("left_label", "")).strip() or "quiet"
    right_label = str(control.get("right_label", "")).strip() or "bright"
    value = control.get("value", 50)
    return [
        {
            "key": _slider_key(label),
            "label": label,
            "left_label": left_label,
            "right_label": right_label,
            "value": value,
        }
    ]


def _slider_from_name(name: str, index: int) -> dict:
    clean = str(name).strip() or f"slider_{index + 1}"
    lower = clean.lower()
    defaults = {
        "density": ("minimal", "layered", 45),
        "intensity": ("soft", "bright", 55),
        "range": ("contained", "open", 50),
        "spacing": ("tight", "open", 50),
        "pulse": ("still", "vibrant", 55),
    }
    left_label, right_label, value = defaults.get(lower, ("less", "more", 50))
    return {
        "key": _slider_key(clean),
        "label": clean.replace("_", " ").title(),
        "left_label": left_label,
        "right_label": right_label,
        "value": value,
    }


def _slider_key(value: str) -> str:
    key = "_".join(str(value).lower().strip().split())
    return "".join(char for char in key if char.isalnum() or char == "_") or "slider"


def _next_need_from_turn(turn: ConversationTurn) -> dict:
    control = turn.controls[0] if turn.controls else None
    return {
        "goal": turn.reason,
        "suggested_component": control.kind.value if control else ControlKind.TEXT_REFINEMENT.value,
        "profile_ready": turn.is_ready,
    }


def _recent_choices(profile: InterviewProfile) -> list[str]:
    answers = profile.ideas + profile.styles + profile.symbols + profile.contrasts + profile.free_notes + profile.offered_options
    answers.extend(str(value) for value in profile.visual_preferences.values() if value not in (None, ""))
    return [_normalize_text(answer)[:80] for answer in answers[-10:] if _normalize_text(answer)]


def _first_nonempty(items: list[str]) -> str:
    for item in items:
        if item:
            return item
    return ""


def sanitize_conversation_turn(
    candidate: ConversationTurn,
    fallback_turn: ConversationTurn,
    profile: InterviewProfile,
) -> ConversationTurn:
    if not candidate.controls:
        raise ValueError("model response did not include a control")
    candidate.controls = [candidate.controls[0]]
    control = candidate.controls[0]

    if candidate.is_ready and not fallback_turn.is_ready:
        candidate.is_ready = False
    if fallback_turn.is_ready:
        candidate.is_ready = True
        candidate.assistant_message = _ready_message(profile)
        candidate.progress_label = "Ready: create the painting"
        candidate.reason = "profile has enough personal signal for generation"
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
        raise ValueError("model requested ready_button before profile was ready")

    expected_kind = _expected_control_kind(fallback_turn)
    if expected_kind and not _control_kind_matches_stage(control.kind, expected_kind):
        raise ValueError(f"model chose {control.kind.value}; expected {expected_kind.value} for this stage")

    if control.kind in {ControlKind.CHOICE_CARDS, ControlKind.MULTI_CHOICE_CARDS, ControlKind.SWATCH_PICKER}:
        control.options = _sanitize_options(control.options, _recent_choices(profile), profile)
    elif control.kind == ControlKind.SLIDER_GROUP:
        control.sliders = _sanitize_sliders(control.sliders)
    elif control.kind == ControlKind.TEXT_REFINEMENT:
        control.options = []
        control.sliders = []

    if control.kind in {ControlKind.CHOICE_CARDS, ControlKind.MULTI_CHOICE_CARDS, ControlKind.SWATCH_PICKER} and not control.options:
        raise ValueError(f"{control.kind.value} needs at least one option")
    if control.kind == ControlKind.SLIDER_GROUP and not control.sliders:
        raise ValueError("slider_group needs at least one slider")
    candidate.assistant_message = _polish_assistant_message(candidate, profile)
    return candidate


def _ready_message(profile: InterviewProfile) -> str:
    idea = _first_nonempty(profile.ideas + profile.free_notes)
    fragment = _short_fragment(idea) if idea else "this feeling"
    return f"I have enough to make {fragment} visible as an abstract painting."


def _polish_assistant_message(candidate: ConversationTurn, profile: InterviewProfile) -> str:
    message = " ".join(candidate.assistant_message.strip().split())
    control = candidate.controls[0] if candidate.controls else None
    normalized = _normalize_text(message).rstrip("?")
    too_thin = len(message.split()) <= 3
    repeated = normalized in {_normalize_text(question).rstrip("?") for question in profile.asked_questions[-5:]}
    main_words = set(_normalize_text(_first_nonempty(profile.ideas)).split())
    overlap = len(main_words.intersection(set(normalized.split()))) >= 2 if main_words else False
    if not control or (not too_thin and not repeated and not overlap):
        return message
    if control.kind == ControlKind.SLIDER_GROUP:
        return "How should that feeling move across the canvas?"
    if control.kind == ControlKind.SWATCH_PICKER:
        return "What color weather should carry this feeling?"
    if control.kind in {ControlKind.CHOICE_CARDS, ControlKind.MULTI_CHOICE_CARDS}:
        if profile.visual_preferences.get("palette_mood"):
            return "What should remain in the room after the first glance?"
        return "Which detail makes this feeling yours?"
    return message or "What should the painting understand next?"


def _expected_control_kind(fallback_turn: ConversationTurn) -> ControlKind | None:
    if not fallback_turn.controls:
        return None
    return fallback_turn.controls[0].kind


def _control_kind_matches_stage(candidate_kind: ControlKind, expected_kind: ControlKind) -> bool:
    if candidate_kind == expected_kind:
        return True
    meaning_kinds = {ControlKind.CHOICE_CARDS, ControlKind.MULTI_CHOICE_CARDS}
    return expected_kind in meaning_kinds and candidate_kind in meaning_kinds


def _sanitize_options(options: list[str], blocked_options: list[str], profile: InterviewProfile) -> list[str]:
    cleaned: list[str] = []
    for option in options:
        value = _clean_option_text(str(option).strip())
        normalized = _normalize_text(value)
        if value and value not in cleaned and normalized not in blocked_options:
            cleaned.append(value)
        if len(cleaned) >= 5:
            break
    if len(cleaned) < 2:
        for option in _companion_options(profile, cleaned):
            normalized = _normalize_text(option)
            if normalized not in blocked_options and option not in cleaned:
                cleaned.append(option)
            if len(cleaned) >= 3:
                break
    return cleaned


def _companion_options(profile: InterviewProfile, existing: list[str]) -> list[str]:
    idea = _normalize_text(_first_nonempty(profile.ideas + profile.free_notes))
    if any(word in idea for word in ["weather", "light", "bright", "joy", "relief", "fresh", "clean"]):
        return ["a silver lift", "clean light", "small bright ripples", "a soft opening"]
    if "lonely" in idea or "loneliness" in idea or "silence" in idea:
        return ["a private echo", "a held breath", "a small warm distance", "an open room"]
    if "positive" in idea or "hope" in idea:
        return ["a small glow", "a rising edge", "a clear morning mark", "a brave softness"]
    if "chaos" in idea:
        return ["a quiet center", "a broken rhythm", "a path through noise", "a bright interruption"]
    return [
        *(existing or []),
        "a hidden second voice",
        "a quiet counterweight",
        "something unnamed",
    ]


def _remember_model_turn(profile: InterviewProfile, turn: ConversationTurn) -> None:
    question = turn.assistant_message.strip()
    if question and question not in profile.asked_questions:
        profile.asked_questions.append(question)
    if turn.controls:
        for option in turn.controls[0].options:
            clean = option.strip()
            if clean and clean not in profile.offered_options:
                profile.offered_options.append(clean)
    profile.asked_questions = profile.asked_questions[-8:]
    profile.offered_options = profile.offered_options[-24:]


def _clean_option_text(value: str) -> str:
    clean = " ".join(value.replace("...", " ").split())
    if not clean:
        return ""
    if len(clean) > 82:
        clean = clean[:79].rstrip(" ,.;:") + "."
    return clean


def _normalize_text(value: str) -> str:
    return " ".join(str(value).lower().strip().split())


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
    companion_dials = [
        SliderSpec(key="space", label="Space", left_label="close", right_label="open", value=50),
        SliderSpec(key="rhythm", label="Rhythm", left_label="still", right_label="moving", value=45),
        SliderSpec(key="edge", label="Edge", left_label="soft", right_label="sharp", value=55),
    ]
    for slider in companion_dials:
        if len(cleaned) >= 3:
            break
        if all(existing.key != slider.key for existing in cleaned):
            cleaned.append(slider)
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
        _apply_choice(profile, values[0] if values else "", response)
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


def _apply_choice(profile: InterviewProfile, value: str, response: ControlResponse | None = None) -> None:
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
    elif _looks_like_model_style_choice(profile, response):
        profile.styles.append(value)
    elif _looks_like_model_symbol_choice(profile, response):
        profile.symbols.append(value)
    elif _looks_like_model_contrast_choice(profile, response):
        profile.contrasts.append(value)
    else:
        profile.free_notes.append(value)


def _looks_like_model_style_choice(profile: InterviewProfile, response: ControlResponse | None) -> bool:
    if response is None or profile.styles:
        return False
    label = _response_context(response)
    return profile.scores.idea_anchor >= 0.65 and (
        profile.scores.visual_taste < 0.45
        or any(word in label for word in ["style", "presence", "mood", "feel", "tone", "texture", "weather"])
    )


def _looks_like_model_symbol_choice(profile: InterviewProfile, response: ControlResponse | None) -> bool:
    if response is None:
        return False
    label = _response_context(response)
    return any(word in label for word in ["symbol", "anchor", "object", "shape", "image", "motif"])


def _looks_like_model_contrast_choice(profile: InterviewProfile, response: ControlResponse | None) -> bool:
    if response is None:
        return False
    label = _response_context(response)
    return any(word in label for word in ["contrast", "pressure", "edge", "tension"])


def _response_context(response: ControlResponse) -> str:
    return _normalize_text(" ".join(response.values + [response.text]))


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
