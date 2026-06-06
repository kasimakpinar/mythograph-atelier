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
) -> tuple[InterviewProfile, ConversationTurn]:
    profile = profile or new_profile()
    clean_message = (user_message or "").strip()

    if clean_message:
        _apply_free_text(profile, clean_message)

    if control_response:
        apply_control_response(profile, control_response)

    profile = update_scores(profile)
    turn = choose_conversation_turn(profile)
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
