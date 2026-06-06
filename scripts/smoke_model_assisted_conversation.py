import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.models.llm_client import LLMResponse
from mythograph.schemas.ui import ControlKind
import mythograph.services.conversation as conversation


class GoodClient:
    def complete_json(self, system_prompt, user_payload):
        return LLMResponse(
            content=json.dumps(
                {
                    "assistant_message": "Choose the visual weather for this private myth.",
                    "progress_label": "Taste: model-authored color weather",
                    "reason": "fake hosted model proposed a palette control",
                    "is_ready": False,
                    "controls": [
                        {
                            "kind": "swatch_picker",
                            "label": "Color weather",
                            "prompt": "Pick the atmosphere.",
                            "options": ["ash, gold, pale blue", "ink, clay, bone"],
                            "sliders": [],
                        }
                    ],
                }
            ),
            source="modal",
        )


class BadClient:
    def complete_json(self, system_prompt, user_payload):
        return LLMResponse(content="not json", source="modal")


def main() -> None:
    original_mode = conversation.CONVERSATION_MODE
    conversation.CONVERSATION_MODE = "model_assisted"
    try:
        profile = conversation.new_profile()
        profile.ideas.append("I want something about patience.")
        profile = conversation.update_scores(profile)
        fallback = conversation.choose_conversation_turn(profile)

        good = conversation.choose_conversation_turn_with_model(profile, fallback, [], GoodClient())
        assert good.controls[0].kind == ControlKind.SWATCH_PICKER
        assert good.controls[0].options[0] == "ash, gold, pale blue"

        bad = conversation.choose_conversation_turn_with_model(profile, fallback, [], BadClient())
        assert bad.model_dump() == fallback.model_dump()
    finally:
        conversation.CONVERSATION_MODE = original_mode

    print("model_assisted_ok")


if __name__ == "__main__":
    main()
