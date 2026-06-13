from mythograph.schemas.profile import InterviewProfile


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
    conversational_ready = (
        profile.turn_count >= 3
        and profile.scores.idea_anchor >= 0.55
        and (bool(profile.symbols) or len(profile.ideas + profile.free_notes) >= 3)
    )
    profile.scores.ready_to_generate = (
        conversational_ready
        or (
            profile.turn_count >= 2
            and profile.scores.idea_anchor >= 0.5
            and profile.scores.visual_taste >= 0.45
            and profile.scores.symbolic_material >= 0.35
        )
    )
    return profile

