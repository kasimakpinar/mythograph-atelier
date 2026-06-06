import time

from mythograph.schemas.profile import InterviewProfile
from mythograph.schemas.ui import NextUI, UIAction
from mythograph.config import ROOT_DIR
from mythograph.models.llm_client import LLMClient, extract_json_object
from mythograph.services.trace_logger import log_event


IDEA_OPTIONS = [
    "Some things are outside my control, but my response is mine.",
    "The best victory is not always fighting harder, but choosing the right position.",
    "I want to become someone stronger than my current habits.",
    "Life can be strange and unfair, but I can still create meaning.",
    "A quiet path can be more powerful than a loud success.",
    "Something old must break so something new can appear.",
]

STYLE_OPTIONS = [
    "quiet and elegant",
    "bold and dramatic",
    "mysterious and symbolic",
    "clean and modern",
    "chaotic but beautiful",
    "surprise me",
]

SYMBOL_OPTIONS = ["a line", "a door", "a flame", "a mountain", "a mirror", "a storm"]

CONTRAST_OPTIONS = [
    "comfort me",
    "challenge me",
    "make it fragile",
    "make it powerful",
]


def new_profile() -> InterviewProfile:
    return InterviewProfile()


def update_scores(profile: InterviewProfile) -> InterviewProfile:
    profile.scores.idea_anchor = min(1.0, 0.35 * len(profile.ideas) + 0.25 * len(profile.free_notes))
    profile.scores.visual_taste = min(
        1.0,
        0.25 * len(profile.styles)
        + 0.18 * sum(1 for value in profile.visual_preferences.values() if value not in (None, "")),
    )
    profile.scores.symbolic_material = min(1.0, 0.3 * len(profile.symbols) + 0.2 * len(profile.contrasts))
    profile.scores.surprise_level = 1.0 if any("surprise" in item.lower() for item in profile.styles + profile.symbols + profile.ideas) else profile.scores.surprise_level
    profile.scores.ready_to_generate = (
        profile.turn_count >= 2
        and profile.scores.idea_anchor >= 0.5
        and profile.scores.visual_taste >= 0.45
        and profile.scores.symbolic_material >= 0.35
    ) or profile.turn_count >= 5
    return profile


def choose_next_ui(profile: InterviewProfile) -> NextUI:
    profile = update_scores(profile)

    if profile.scores.ready_to_generate:
        return NextUI(
            assistant_message="I have enough to turn this into symbols, colors, and motion.",
            next_action=UIAction.READY_TO_GENERATE,
            reason="core objectives are sufficiently filled",
            question="Ready to create the painting?",
            options=["Create artwork"],
        )

    if profile.scores.idea_anchor < 0.5:
        return NextUI(
            assistant_message="Let us give the painting a real idea to hold onto.",
            next_action=UIAction.SHOW_IDEA_CARDS,
            reason="idea_anchor is weak",
            question="Pick one or two ideas that feel interesting today.",
            options=IDEA_OPTIONS,
        )

    if profile.scores.visual_taste < 0.45:
        return NextUI(
            assistant_message="Now I need to learn what kind of abstract image you would actually enjoy looking at.",
            next_action=UIAction.SHOW_VISUAL_SLIDERS,
            reason="visual_taste is weak",
            question="Shape the visual language.",
            options=[],
        )

    if not profile.styles:
        return NextUI(
            assistant_message="The taste is forming. Choose a direction for the overall presence of the piece.",
            next_action=UIAction.SHOW_STYLE_CARDS,
            reason="style language is missing",
            question="Which aesthetic direction should lead?",
            options=STYLE_OPTIONS,
        )

    if profile.scores.symbolic_material < 0.35:
        return NextUI(
            assistant_message="I need one image that can become the heart of the painting.",
            next_action=UIAction.SHOW_SYMBOL_CARDS,
            reason="symbolic_material is weak",
            question="Which symbol should carry the meaning?",
            options=SYMBOL_OPTIONS,
        )

    return NextUI(
        assistant_message="One last choice will decide the emotional angle.",
        next_action=UIAction.ASK_CONTRAST,
        reason="a decisive contrast would sharpen the recipe",
        question="Should the painting comfort you or challenge you?",
        options=CONTRAST_OPTIONS,
    )


def choose_next_ui_with_model(profile: InterviewProfile, client: LLMClient | None = None) -> NextUI:
    fallback = choose_next_ui(profile)
    llm = client or LLMClient()
    system_prompt = (ROOT_DIR / "mythograph" / "prompts" / "interviewer_system.txt").read_text(encoding="utf-8")
    started = time.perf_counter()
    response = llm.complete_json(
        system_prompt,
        {
            "profile": profile.model_dump(),
            "objective_scores": profile.scores.model_dump(),
            "fallback_next_ui": fallback.model_dump(),
            "allowed_actions": [action.value for action in UIAction],
        },
    )
    elapsed_seconds = round(time.perf_counter() - started, 3)

    if response.source == "mock":
        log_event(
            "llm_ui_director",
            {
                "source": "mock",
                "elapsed_seconds": elapsed_seconds,
                "used_fallback": True,
                "next_ui": fallback.model_dump(),
            },
        )
        return fallback

    if response.error:
        log_event(
            "llm_ui_director",
            {
                "source": response.source,
                "elapsed_seconds": elapsed_seconds,
                "error": response.error,
                "transport_error": response.error,
                "raw_content": response.content,
                "used_fallback": True,
                "next_ui": fallback.model_dump(),
            },
        )
        return fallback

    try:
        candidate = NextUI.model_validate(extract_json_object(response.content))
    except Exception as exc:
        log_event(
            "llm_ui_director",
            {
                "source": response.source,
                "elapsed_seconds": elapsed_seconds,
                "error": str(exc),
                "transport_error": response.error,
                "raw_content": response.content,
                "used_fallback": True,
                "next_ui": fallback.model_dump(),
            },
        )
        return fallback

    candidate = _sanitize_next_ui(candidate, fallback)
    log_event(
        "llm_ui_director",
        {
            "source": response.source,
            "elapsed_seconds": elapsed_seconds,
            "transport_error": response.error,
            "raw_content": response.content,
            "used_fallback": False,
            "next_ui": candidate.model_dump(),
        },
    )
    return candidate


def _sanitize_next_ui(candidate: NextUI, fallback: NextUI) -> NextUI:
    if candidate.next_action == UIAction.READY_TO_GENERATE and fallback.next_action != UIAction.READY_TO_GENERATE:
        return fallback
    if candidate.next_action == UIAction.SHOW_VISUAL_SLIDERS:
        candidate.options = []
    if candidate.next_action in {
        UIAction.SHOW_IDEA_CARDS,
        UIAction.SHOW_STYLE_CARDS,
        UIAction.SHOW_SYMBOL_CARDS,
        UIAction.ASK_CONTRAST,
    } and not candidate.options:
        candidate.options = fallback.options
    return candidate


def apply_answer(profile: InterviewProfile, action: str, answer: str, visual_values: dict[str, float] | None = None) -> InterviewProfile:
    clean_answer = (answer or "").strip()
    if clean_answer:
        if action == UIAction.SHOW_IDEA_CARDS:
            profile.ideas.append(clean_answer)
        elif action == UIAction.SHOW_STYLE_CARDS:
            profile.styles.append(clean_answer)
        elif action == UIAction.SHOW_SYMBOL_CARDS:
            profile.symbols.append(clean_answer)
        elif action == UIAction.ASK_CONTRAST:
            profile.contrasts.append(clean_answer)
        elif action == UIAction.SURPRISE_STEP:
            profile.styles.append("surprise me")
            profile.symbols.append(clean_answer)
        else:
            profile.free_notes.append(clean_answer)

    if visual_values:
        profile.visual_preferences.update(visual_values)

    profile.turn_count += 1
    return update_scores(profile)


def start_with_surprise(profile: InterviewProfile) -> InterviewProfile:
    profile.ideas.append("Life can be strange and unfair, but I can still create meaning.")
    profile.styles.append("mysterious and symbolic")
    profile.symbols.append("a mirror")
    profile.scores.surprise_level = 1.0
    profile.turn_count += 1
    return update_scores(profile)
