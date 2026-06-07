from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.schemas.ui import ControlKind, ControlResponse
import mythograph.services.conversation as conversation


def main() -> None:
    original_mode = conversation.CONVERSATION_MODE
    conversation.CONVERSATION_MODE = "deterministic"
    try:
        turn = conversation.start_session()
        profile, turn = conversation.advance_conversation(
            conversation.new_profile(),
            user_message="I want something about patience and ambition.",
        )
        assert turn.controls, "conversation should ask for a dynamic control"

        profile, turn = conversation.advance_conversation(
            profile,
            control_response=ControlResponse(
                kind=ControlKind.MULTI_CHOICE_CARDS,
                values=["A quiet path can be more powerful than a loud success."],
            ),
        )
        profile, turn = conversation.advance_conversation(
            profile,
            control_response=ControlResponse(
                kind=ControlKind.SLIDER_GROUP,
                sliders={"minimal_rich": 30, "calm_intense": 70, "geometric_organic": 55},
            ),
        )
        profile, turn = conversation.advance_conversation(
            profile,
            control_response=ControlResponse(kind=ControlKind.SWATCH_PICKER, values=["bone, ink, muted gold"]),
        )
        profile, turn = conversation.advance_conversation(
            profile,
            control_response=ControlResponse(kind=ControlKind.CHOICE_CARDS, values=["mysterious and symbolic"]),
        )
        profile, turn = conversation.advance_conversation(
            profile,
            control_response=ControlResponse(kind=ControlKind.MULTI_CHOICE_CARDS, values=["a mountain", "a flame"]),
        )
    finally:
        conversation.CONVERSATION_MODE = original_mode

    assert conversation.should_generate(profile), "profile should be ready within the compact interview"
    assert turn.is_ready, "latest turn should expose the ready button"
    print(turn.controls[0].kind)
    print(profile.scores.ready_to_generate)


if __name__ == "__main__":
    main()
