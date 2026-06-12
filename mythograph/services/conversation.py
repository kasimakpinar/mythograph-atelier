import re
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
    if clean_message and _user_requested_generation(clean_message) and _model_can_generate(profile):
        turn = _ready_turn()
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
    can_generate = _model_can_generate(profile)
    task = _conversation_task(profile, can_generate)
    allowed_control_kinds = _allowed_control_kinds(fallback_turn, can_generate, task)
    payload = {
        "task": task,
        "atelier_state": build_atelier_state(profile),
        "next_need": _next_need_for_model(profile, fallback_turn, task),
        "can_generate": can_generate,
        "conversation_budget": "Aim for a useful artwork brief in 3-5 meaningful exchanges. Keep chatting when the user's own words matter; use controls only when they help.",
        "controls_are_optional": True,
        "allowed_control_kinds": allowed_control_kinds,
        "control_guidance": {
            "empty_controls": "Use plain chat when a real question is better than buttons.",
            "choice_cards": "3-6 interpretations, emotional angles, stances, or readings.",
            "multi_choice_cards": "3-6 compatible tensions, values, memories, or admissions; user may pick several.",
            "slider_group": "2-3 conceptual dials such as honesty, control, distance, tenderness, acceptance, resistance.",
            "swatch_picker": "rare mood climates, not literal color settings.",
            "text_refinement": "invite a free reply in the main chat when the topic needs the user's own words.",
            "ready_button": "when can_generate is true and the profile has enough personal signal.",
        },
    }
    response = llm.complete_json(
        system_prompt,
        payload,
        max_tokens=max(LLM_CHAT_MAX_TOKENS, 180),
        response_format={"type": "json_object"},
    )

    if response.source == "mock":
        return fallback_turn

    last_error = response.error or ""
    last_content = response.content
    current_response = response
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            if current_response.error:
                raise ValueError(current_response.error)
            candidate = ConversationTurn.model_validate(
                _normalize_turn_payload(extract_json_object(current_response.content), fallback_turn)
            )
            candidate = sanitize_conversation_turn(candidate, fallback_turn, profile, can_generate, allowed_control_kinds)
            quality_issue = _turn_quality_issue(candidate, profile)
            if quality_issue:
                raise ValueError(quality_issue)
        except Exception as exc:
            last_error = str(exc)
            last_content = current_response.content or last_content
            if attempt >= max_attempts - 1:
                elapsed_seconds = round(time.perf_counter() - started, 3)
                error_turn = safe_conversation_fallback(profile, last_error, can_generate)
                log_event(
                    "llm_conversation_turn",
                    {
                        "source": current_response.source or response.source,
                        "elapsed_seconds": elapsed_seconds,
                        "error": last_error,
                        "transport_error": current_response.error,
                        "raw_content": last_content,
                        "raw_preview": _preview(last_content),
                        "used_fallback": False,
                        "retry_count": attempt,
                        "chosen_control": error_turn.controls[0].kind if error_turn.controls else None,
                        "atelier_state": build_atelier_state(profile),
                        "turn": error_turn.model_dump(),
                    },
                )
                return error_turn

            current_response = llm.complete_json(
                _repair_system_prompt(),
                {
                    "invalid_json": current_response.content,
                    "previous_error": last_error,
                    "attempt": attempt + 1,
                    "required_shape": "ConversationTurn JSON with assistant_message, progress_label, reason, is_ready, controls.",
                    "next_need": _next_need_for_model(profile, fallback_turn, task),
                    "can_generate": can_generate,
                    "allowed_control_kinds": allowed_control_kinds,
                    "atelier_state": build_atelier_state(profile),
                    "quality_rules": _quality_rules_for_retry(profile),
                },
                max_tokens=max(LLM_CHAT_MAX_TOKENS, 180),
                response_format={"type": "json_object"},
            )
            continue

        elapsed_seconds = round(time.perf_counter() - started, 3)
        log_event(
            "llm_conversation_turn",
            {
                "source": current_response.source,
                "elapsed_seconds": elapsed_seconds,
                "error": last_error if attempt else None,
                "transport_error": current_response.error,
                "raw_content": current_response.content,
                "raw_preview": _preview(current_response.content),
                "used_fallback": False,
                "retry_count": attempt,
                "chosen_control": candidate.controls[0].kind if candidate.controls else None,
                "atelier_state": build_atelier_state(profile),
                "turn": candidate.model_dump(),
            },
        )
        return candidate

    return safe_conversation_fallback(profile, last_error)


def model_error_turn(message: str, detail: str = "") -> ConversationTurn:
    reason = "llama.cpp did not return valid schema JSON"
    if detail:
        reason = f"{reason}: {detail[:900]}"
    return ConversationTurn(
        assistant_message=message,
        progress_label="Understanding your theme",
        reason=reason,
        is_ready=False,
        controls=[],
    )


def safe_conversation_fallback(
    profile: InterviewProfile,
    detail: str = "",
    can_generate: bool | None = None,
) -> ConversationTurn:
    if can_generate is None:
        can_generate = _model_can_generate(profile)
    if can_generate:
        reason = "Fallback after invalid model output."
        if detail:
            reason = f"{reason} {detail[:600]}"
        return _ready_turn(reason)

    idea = _first_nonempty(profile.ideas + profile.free_notes)
    if profile.turn_count <= 2:
        message = (
            "That helps. Give me one concrete place where this shows up in real life: "
            "work, relationships, money, time, ambition, family, or something else."
        )
    elif not profile.contrasts:
        message = (
            "I think I understand the theme better now. What attitude should the painting hold toward it: "
            "acceptance, resistance, trust, doubt, tenderness, or something else in your own words?"
        )
    else:
        message = "I think I have enough of the meaning now. Add one last correction if I missed the heart of it."
    if not idea:
        message = "Tell me what this should be about in ordinary words. One sentence is enough."
    reason = "Fallback after invalid model output."
    if detail:
        reason = f"{reason} {detail[:600]}"
    return ConversationTurn(
        assistant_message=message,
        progress_label="Understanding your theme",
        reason=reason,
        is_ready=False,
        controls=[],
    )


def _ready_turn(reason: str = "profile has enough personal signal for generation") -> ConversationTurn:
    return ConversationTurn(
        assistant_message=_ready_message(),
        progress_label="Ready: create the painting",
        reason=reason,
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


def _preview(text: str, limit: int = 900) -> str:
    return " ".join((text or "").split())[:limit]


def _repair_system_prompt() -> str:
    return (
        "Return valid JSON only for a Mythograph Atelier ConversationTurn. "
        "Use the schema fields assistant_message, progress_label, reason, is_ready, controls. "
        "Controls may be empty. If a control is useful, include one control using an allowed kind name. "
        "Ask about the user's theme, memory, quote, value, contradiction, or belief in plain language. "
        "The control is semantic only; the custom frontend handles layout."
    )


def _quality_rules_for_retry(profile: InterviewProfile) -> dict:
    return {
        "avoid_questions": profile.asked_questions[-8:],
        "avoid_options": profile.offered_options[-24:],
        "minimum_options_for_card_controls": 3,
        "controls_are_optional": True,
        "prefer_different_control_kind_than_recent": profile.control_history[-3:],
    }


def _turn_quality_issue(candidate: ConversationTurn, profile: InterviewProfile) -> str:
    message = _normalize_text(candidate.assistant_message)
    if any(_texts_too_similar(message, _normalize_text(question)) for question in profile.asked_questions[-6:]):
        return "assistant_message repeats a previous question"

    control = candidate.controls[0] if candidate.controls else None
    if not control or candidate.is_ready:
        return ""

    prompt = _normalize_text(control.prompt)
    if any(_texts_too_similar(prompt, _normalize_text(question)) for question in profile.asked_questions[-6:]):
        return "control prompt repeats a previous question"

    if control.kind in {ControlKind.CHOICE_CARDS, ControlKind.MULTI_CHOICE_CARDS}:
        if len(control.options) < 3:
            return f"{control.kind.value} needs at least three useful options"
        if _options_too_similar(control.options, profile.offered_options[-24:]):
            return "options are too similar to previous options"

    if control.kind == ControlKind.TEXT_REFINEMENT and not _looks_like_question(candidate.assistant_message, control.prompt):
        return "text_refinement must ask a clear question"

    recent_kinds = profile.control_history[-3:]
    if len(recent_kinds) >= 2 and all(kind == control.kind.value for kind in recent_kinds[-2:]):
        return f"control kind {control.kind.value} has repeated too often"

    return ""


def _looks_like_question(message: str, prompt: str) -> bool:
    text = f"{message} {prompt}".strip()
    if "?" in text:
        return True
    lowered = _normalize_text(text)
    starts = ("what ", "which ", "when ", "where ", "who ", "whose ", "how ", "why ", "tell me ", "describe ")
    return lowered.startswith(starts) or any(f" {start}" in lowered for start in starts)


def _texts_too_similar(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = _meaning_tokens(left)
    right_tokens = _meaning_tokens(right)
    if len(left_tokens) < 4 or len(right_tokens) < 4:
        return False
    if _shared_ngram(left_tokens, right_tokens, 6):
        return True
    left_words = set(left_tokens)
    right_words = set(right_tokens)
    overlap = len(left_words & right_words) / max(1, min(len(left_words), len(right_words)))
    return overlap >= 0.62


def _meaning_tokens(value: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9']+", value.lower()) if len(word) > 3]


def _shared_ngram(left: list[str], right: list[str], size: int) -> bool:
    if len(left) < size or len(right) < size:
        return False
    left_grams = {tuple(left[index : index + size]) for index in range(len(left) - size + 1)}
    return any(tuple(right[index : index + size]) in left_grams for index in range(len(right) - size + 1))


def _options_too_similar(options: list[str], previous_options: list[str]) -> bool:
    previous = [_normalize_text(option) for option in previous_options]
    repeated = 0
    for option in options:
        normalized = _normalize_text(option)
        if normalized in previous or any(_texts_too_similar(normalized, item) for item in previous):
            repeated += 1
    return repeated >= max(2, len(options) // 2 + 1)


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
        "avoid_phrases": [
            "one more ingredient",
            "I need one more thing",
            "before the image can settle",
            "choose the anchor",
            "what visual language fits",
            "what should stay unresolved",
        ],
        "recent_control_kinds": _recent_control_kinds(profile),
    }


def _recent_control_kinds(profile: InterviewProfile) -> list[str]:
    return profile.control_history[-4:]


def _normalize_turn_payload(payload: dict, fallback_turn: ConversationTurn) -> dict:
    if not isinstance(payload, dict):
        return payload
    payload["assistant_message"] = str(payload.get("assistant_message", "")).strip() or fallback_turn.assistant_message
    payload["progress_label"] = str(payload.get("progress_label", "")).strip() or _fallback_progress_label(payload, fallback_turn)
    payload["reason"] = str(payload.get("reason", "")).strip() or "model chose the next atelier move"
    controls = payload.get("controls")
    if not isinstance(controls, list) or not controls:
        return payload

    control = controls[0]
    if not isinstance(control, dict):
        return payload

    control["label"] = str(control.get("label", "")).strip() or _label_from_kind(control.get("kind"))
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
        ControlKind.READY_BUTTON.value,
    }:
        control["sliders"] = []
        if not isinstance(control.get("options"), list):
            control["options"] = []
    controls[0] = control
    payload["controls"] = [control]
    return payload


def _fallback_progress_label(payload: dict, fallback_turn: ConversationTurn) -> str:
    if payload.get("is_ready"):
        return "Ready"
    if fallback_turn.progress_label:
        return f"Listening: {fallback_turn.progress_label.split(':', 1)[0].lower()}"
    return "Listening"


def _label_from_kind(kind: object) -> str:
    labels = {
        ControlKind.CHOICE_CARDS.value: "Choose one",
        ControlKind.MULTI_CHOICE_CARDS.value: "Choose what fits",
        ControlKind.SLIDER_GROUP.value: "Tune the meaning",
        ControlKind.SWATCH_PICKER.value: "Choose atmosphere",
        ControlKind.TEXT_REFINEMENT.value: "Your words",
        ControlKind.READY_BUTTON.value: "Create artwork",
    }
    return labels.get(str(kind), "Atelier choice")


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
        return [_normalize_slider_dict(slider, index) for index, slider in enumerate(sliders[:3])]

    if isinstance(sliders, list) and sliders and all(isinstance(slider, str) for slider in sliders):
        return [_slider_from_name(name, index) for index, name in enumerate(sliders[:3])]

    label = str(control.get("label", "")).strip() or "Intensity"
    left_label = str(control.get("left_label", "")).strip() or "quiet"
    right_label = str(control.get("right_label", "")).strip() or "bright"
    value = control.get("value", 50)
    return [
        _normalize_slider_dict({
            "key": _slider_key(label),
            "label": label,
            "left_label": left_label,
            "right_label": right_label,
            "value": value,
        }, 0)
    ]


def _normalize_slider_dict(slider: dict, index: int) -> dict:
    label = str(slider.get("label", "")).strip() or str(slider.get("key", "")).strip() or f"Slider {index + 1}"
    key = str(slider.get("key", "")).strip() or _slider_key(label)
    return {
        "key": _slider_key(key),
        "label": label,
        "left_label": str(slider.get("left_label", "")).strip() or "less",
        "right_label": str(slider.get("right_label", "")).strip() or "more",
        "value": int(float(slider.get("value", 50) or 50)),
    }


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


def _conversation_task(profile: InterviewProfile, can_generate: bool) -> str:
    if can_generate:
        return "conversation_with_ui"
    recent_controls = profile.control_history[-2:]
    if profile.turn_count <= 1:
        return "conversation_chat"
    if len(recent_controls) >= 2:
        return "conversation_chat"
    if _user_seems_unsure(profile) or profile.turn_count % 3 == 0:
        return "conversation_with_ui"
    return "conversation_chat"


def _user_seems_unsure(profile: InterviewProfile) -> bool:
    text = " ".join(profile.ideas + profile.free_notes).lower()
    markers = ["not sure", "don't know", "cannot explain", "can't explain", "maybe", "or ", "but also"]
    return any(marker in text for marker in markers)


def _next_need_for_model(profile: InterviewProfile, turn: ConversationTurn, task: str) -> dict:
    should_use_ui = task == "conversation_with_ui"
    return {
        "goal": _open_interview_target(profile, turn),
        "reason": _open_interview_reason(profile, turn),
        "should_use_ui": should_use_ui,
        "preferred_control_kind": ControlKind.READY_BUTTON.value if _model_can_generate(profile) else None,
        "missing_signals": _missing_signals(profile),
        "director_note": (
            "Use plain chat unless a semantic UI control would genuinely help the user choose, compare, or continue."
        ),
    }


def _model_can_generate(profile: InterviewProfile) -> bool:
    has_personal_meaning = profile.scores.idea_anchor >= 0.55
    has_interpretive_signal = bool(profile.contrasts) or bool(profile.styles) or bool(profile.visual_preferences)
    has_personal_detail = bool(profile.symbols) or len(profile.ideas + profile.free_notes) >= 3
    return profile.turn_count >= 3 and has_personal_meaning and has_interpretive_signal and has_personal_detail


def _missing_signals(profile: InterviewProfile) -> list[str]:
    missing: list[str] = []
    if profile.scores.idea_anchor < 0.55:
        missing.append("personal meaning in the user's own words")
    if not profile.symbols and len(profile.ideas + profile.free_notes) < 3:
        missing.append("one personal memory, example, quote interpretation, contradiction, or metaphor")
    if profile.scores.visual_taste < 0.18 and not profile.styles and not profile.visual_preferences:
        missing.append("one interpretive stance such as acceptance, resistance, tenderness, courage, grief, control, or release")
    if profile.turn_count < 3:
        missing.append("another exchange before generation")
    return missing


def _open_interview_target(profile: InterviewProfile, turn: ConversationTurn) -> str:
    missing = _missing_signals(profile)
    if not missing:
        return "decide whether to generate or ask one high-value follow-up"
    return missing[0]


def _open_interview_reason(profile: InterviewProfile, turn: ConversationTurn) -> str:
    missing = _missing_signals(profile)
    if missing:
        return f"The artwork brief still needs {missing[0]}."
    return "The profile has enough signal; only continue if a sharper connection is worth it."


def _target_from_turn(turn: ConversationTurn) -> str:
    if turn.is_ready:
        return "readiness"
    label = f"{turn.progress_label} {turn.reason}".lower()
    if "palette" in label or "color" in label:
        return "palette_mood"
    if "symbol" in label or "anchor" in label:
        return "main_symbol"
    if "style" in label or "presence" in label:
        return "visual_style"
    if "taste" in label or "visual" in label:
        return "visual_preferences"
    if "mood" in label:
        return "mood"
    return "main_idea"


def _allowed_control_kinds(
    fallback_turn: ConversationTurn,
    can_generate: bool = False,
    task: str = "conversation_with_ui",
) -> list[str]:
    if task == "conversation_chat" and not can_generate:
        return []
    kinds = [
        ControlKind.TEXT_REFINEMENT.value,
        ControlKind.CHOICE_CARDS.value,
        ControlKind.MULTI_CHOICE_CARDS.value,
        ControlKind.SLIDER_GROUP.value,
        ControlKind.SWATCH_PICKER.value,
    ]
    if can_generate:
        kinds.append(ControlKind.READY_BUTTON.value)
    return kinds


def _recent_choices(profile: InterviewProfile) -> list[str]:
    answers = profile.ideas + profile.styles + profile.symbols + profile.contrasts + profile.free_notes + profile.offered_options
    answers.extend(str(value) for value in profile.visual_preferences.values() if value not in (None, ""))
    return [_normalize_text(answer)[:80] for answer in answers[-10:] if _normalize_text(answer)]


def _user_requested_generation(message: str) -> bool:
    text = _normalize_text(message)
    intent_words = ("create", "generate", "make", "paint", "render")
    object_words = ("image", "artwork", "painting", "piece", "art")
    if any(phrase in text for phrase in ["you can create", "create it now", "generate it now", "ready to create"]):
        return True
    return any(word in text for word in intent_words) and any(word in text for word in object_words)


def _first_nonempty(items: list[str]) -> str:
    for item in items:
        if item:
            return item
    return ""


def sanitize_conversation_turn(
    candidate: ConversationTurn,
    fallback_turn: ConversationTurn,
    profile: InterviewProfile,
    can_generate: bool | None = None,
    allowed_control_kinds: list[str] | None = None,
) -> ConversationTurn:
    if can_generate is None:
        can_generate = fallback_turn.is_ready

    if not candidate.controls:
        if candidate.is_ready and can_generate:
            candidate.controls = [
                DynamicControl(
                    kind=ControlKind.READY_BUTTON,
                    label="Create artwork",
                    prompt="Generate the painting",
                    options=["Create artwork"],
                )
            ]
        else:
            candidate.is_ready = False
            candidate.assistant_message = _polish_assistant_message(candidate, profile)
            return candidate

    candidate.controls = [candidate.controls[0]]
    control = candidate.controls[0]
    allowed_control_kinds = allowed_control_kinds or []
    if allowed_control_kinds and control.kind.value not in allowed_control_kinds:
        raise ValueError(f"control kind {control.kind.value} was not allowed for this task")
    if not allowed_control_kinds and control.kind != ControlKind.READY_BUTTON:
        raise ValueError(f"task expected chat-only response but got {control.kind.value}")

    if control.kind != ControlKind.READY_BUTTON:
        candidate.is_ready = False

    if control.kind == ControlKind.READY_BUTTON and can_generate:
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

    if control.kind == ControlKind.READY_BUTTON and not can_generate:
        raise ValueError("model requested ready_button before profile was ready")

    if control.kind in {ControlKind.CHOICE_CARDS, ControlKind.MULTI_CHOICE_CARDS, ControlKind.SWATCH_PICKER}:
        if control.kind in {ControlKind.CHOICE_CARDS, ControlKind.MULTI_CHOICE_CARDS} and len(control.options) < 3:
            raise ValueError(f"{control.kind.value} must provide at least three options before sanitization")
        control.options = _sanitize_options(control.options, _recent_choices(profile), profile)
        control.prompt = _sanitize_control_prompt(control.prompt, control.kind)
    elif control.kind == ControlKind.SLIDER_GROUP:
        control.sliders = _sanitize_sliders(control.sliders)
    elif control.kind == ControlKind.TEXT_REFINEMENT:
        control.options = []
        control.sliders = []

    if control.kind in {ControlKind.CHOICE_CARDS, ControlKind.MULTI_CHOICE_CARDS, ControlKind.SWATCH_PICKER} and len(control.options) < 3:
        raise ValueError(f"{control.kind.value} needs at least three options")
    if control.kind == ControlKind.SLIDER_GROUP and not control.sliders:
        raise ValueError("slider_group needs at least one slider")
    candidate.assistant_message = _polish_assistant_message(candidate, profile)
    return candidate


def _ready_message(profile: InterviewProfile | None = None) -> str:
    return (
        "I have enough now. I will turn what you shared into a simple abstract painting recipe, "
        "then render it as an image."
    )


def _polish_assistant_message(candidate: ConversationTurn, profile: InterviewProfile) -> str:
    message = " ".join(candidate.assistant_message.strip().split())
    control = candidate.controls[0] if candidate.controls else None
    normalized = _normalize_text(message).rstrip("?")
    too_thin = len(message.split()) <= 3 or len(message) < 22
    repeated = normalized in {_normalize_text(question).rstrip("?") for question in profile.asked_questions[-5:]}
    if not control or (not too_thin and not repeated):
        return message
    if control.kind == ControlKind.SLIDER_GROUP:
        return "Which inner tension should lead this meaning?"
    if control.kind == ControlKind.SWATCH_PICKER:
        return "Which atmosphere should carry this meaning?"
    if control.kind in {ControlKind.CHOICE_CARDS, ControlKind.MULTI_CHOICE_CARDS}:
        return "Which interpretation feels closest to what you mean?"
    return message or "What should the atelier understand next?"


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
    if len(cleaned) < 3:
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
        return ["relief after heaviness", "permission to begin again", "joy that feels almost private", "a bright feeling with a quiet cost"]
    if "lonely" in idea or "loneliness" in idea or "silence" in idea:
        return ["peace inside solitude", "wanting distance without disappearing", "being alone but still answered", "softness without rescue"]
    if "positive" in idea or "hope" in idea:
        return ["hope without certainty", "choosing the next step anyway", "quiet optimism", "strength that stays gentle"]
    if "chaos" in idea:
        return ["calm as resistance", "control without stiffness", "not running from noise", "a private center under pressure"]
    return [
        *(existing or []),
        "acceptance without surrender",
        "a feeling that has not found words yet",
        "tenderness under pressure",
    ]


def _remember_model_turn(profile: InterviewProfile, turn: ConversationTurn) -> None:
    question = turn.assistant_message.strip()
    if question and question not in profile.asked_questions:
        profile.asked_questions.append(question)
    if turn.controls:
        kind = turn.controls[0].kind.value
        if kind not in {ControlKind.READY_BUTTON.value, ControlKind.TEXT_REFINEMENT.value}:
            profile.control_history.append(kind)
        for option in turn.controls[0].options:
            clean = option.strip()
            if clean and clean not in profile.offered_options:
                profile.offered_options.append(clean)
    profile.asked_questions = profile.asked_questions[-8:]
    profile.offered_options = profile.offered_options[-24:]
    profile.control_history = profile.control_history[-8:]


def _clean_option_text(value: str) -> str:
    clean = " ".join(value.replace("...", " ").split())
    if not clean:
        return ""
    if len(clean) > 120:
        boundary = clean.rfind(" ", 0, 118)
        cut_at = boundary if boundary >= 70 else 117
        clean = clean[:cut_at].rstrip(" ,.;:") + "."
    return clean


def _sanitize_control_prompt(prompt: str, kind: ControlKind) -> str:
    clean = " ".join((prompt or "").strip().split())
    if kind == ControlKind.CHOICE_CARDS:
        replacements = {
            "or choose a few if none fit perfectly": "or keep typing if none fit perfectly",
            "or pick a few if none fit perfectly": "or keep typing if none fit perfectly",
            "pick any that feel true": "pick the closest one, or keep typing",
            "choose any that feel true": "choose the closest one, or keep typing",
        }
        lowered = clean.lower()
        for old, new in replacements.items():
            if old in lowered:
                start = lowered.find(old)
                clean = clean[:start] + new + clean[start + len(old) :]
                lowered = clean.lower()
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
        elif _looks_like_model_symbol_choice(profile, response):
            profile.symbols.extend(values)
        else:
            profile.ideas.extend(values)
    elif response.kind == ControlKind.CHOICE_CARDS:
        _apply_choice(profile, values[0] if values else "", response)
    elif response.kind == ControlKind.SLIDER_GROUP:
        profile.visual_preferences.update(response.sliders)
        if "meaning dials" not in profile.free_notes:
            profile.free_notes.append("meaning dials")
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
    elif _looks_like_model_interpretive_choice(response):
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
    if _looks_like_model_interpretive_choice(response):
        return False
    return profile.scores.idea_anchor >= 0.65 and (
        profile.scores.visual_taste < 0.45
        or any(word in label for word in ["style", "presence", "mood", "feel", "tone", "texture", "weather"])
    )


def _looks_like_model_interpretive_choice(response: ControlResponse | None) -> bool:
    if response is None:
        return False
    label = _response_context(response)
    return any(
        word in label
        for word in [
            "stance",
            "interpretation",
            "reading",
            "meaning",
            "attitude",
            "belief",
            "value",
            "admission",
        ]
    )


def _looks_like_model_symbol_choice(profile: InterviewProfile, response: ControlResponse | None) -> bool:
    if response is None:
        return False
    label = _response_context(response)
    if any(word in label for word in ["symbol", "anchor", "object", "shape", "image", "motif", "ingredient"]):
        return True
    values = [value.strip().lower() for value in response.values if value.strip()]
    concrete_noun_phrase = any(value.startswith(("a ", "an ", "the ")) for value in values)
    return profile.scores.idea_anchor >= 0.5 and concrete_noun_phrase and not profile.symbols


def _looks_like_model_contrast_choice(profile: InterviewProfile, response: ControlResponse | None) -> bool:
    if response is None:
        return False
    label = _response_context(response)
    return any(word in label for word in ["contrast", "pressure", "edge", "tension"])


def _response_context(response: ControlResponse) -> str:
    return _normalize_text(" ".join(response.values + [response.text, response.label, response.prompt]))


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
