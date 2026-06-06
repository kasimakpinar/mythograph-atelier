from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.schemas.ui import ControlKind, ControlResponse
from mythograph.services.conversation import advance_conversation, new_profile, should_generate, start_session


def main() -> None:
    turn = start_session()
    profile, turn = advance_conversation(new_profile(), user_message="I want something about patience and ambition.")
    assert turn.controls, "conversation should ask for a dynamic control"

    profile, turn = advance_conversation(
        profile,
        control_response=ControlResponse(
            kind=ControlKind.MULTI_CHOICE_CARDS,
            values=["A quiet path can be more powerful than a loud success."],
        ),
    )
    profile, turn = advance_conversation(
        profile,
        control_response=ControlResponse(
            kind=ControlKind.SLIDER_GROUP,
            sliders={"minimal_rich": 30, "calm_intense": 70, "geometric_organic": 55},
        ),
    )
    profile, turn = advance_conversation(
        profile,
        control_response=ControlResponse(kind=ControlKind.SWATCH_PICKER, values=["bone, ink, muted gold"]),
    )
    profile, turn = advance_conversation(
        profile,
        control_response=ControlResponse(kind=ControlKind.CHOICE_CARDS, values=["mysterious and symbolic"]),
    )
    profile, turn = advance_conversation(
        profile,
        control_response=ControlResponse(kind=ControlKind.MULTI_CHOICE_CARDS, values=["a mountain", "a flame"]),
    )

    assert should_generate(profile), "profile should be ready within the compact interview"
    assert turn.is_ready, "latest turn should expose the ready button"
    print(turn.controls[0].kind)
    print(profile.scores.ready_to_generate)


if __name__ == "__main__":
    main()
