import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.models.llm_client import LLMResponse
from mythograph.schemas.ui import ControlKind
import mythograph.services.conversation as conversation


class GoodClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None):
        assert max_tokens == conversation.LLM_CHAT_MAX_TOKENS
        assert response_format == {"type": "json_object"}
        if "invalid_json" in user_payload:
            return LLMResponse(content=user_payload["invalid_json"], source="llamacpp")
        assert "atelier_state" in user_payload
        assert "chat_history" not in user_payload
        assert "fallback_turn" not in user_payload
        assert "available_option_sets" not in user_payload
        assert "next_need" in user_payload
        assert "can_generate" in user_payload
        assert len(user_payload["atelier_state"]["answers_so_far"]) <= 6
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
            source="llamacpp",
        )


class BadClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None):
        return LLMResponse(content="not json", source="llamacpp")


class RepairClient:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="not json", source="llamacpp")
        return LLMResponse(
            content=json.dumps(
                {
                    "assistant_message": "Choose the symbol that keeps returning.",
                    "progress_label": "Symbol: repaired",
                    "reason": "repair produced valid UI JSON",
                    "is_ready": False,
                    "controls": [
                        {
                            "kind": "swatch_picker",
                            "label": "Quiet weather",
                            "prompt": "Pick one atmosphere.",
                            "options": ["rain glass, graphite, low amber", "chalk, moss, soft black"],
                            "sliders": [],
                        }
                    ],
                }
            ),
            source="llamacpp",
        )


class WrongStageClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None):
        return LLMResponse(
            content=json.dumps(
                {
                    "assistant_message": "How does patience feel?",
                    "progress_label": "Looping",
                    "reason": "wrong component",
                    "is_ready": False,
                    "controls": [
                        {
                            "kind": "choice_cards",
                            "label": "Feel",
                            "prompt": "Choose one.",
                            "options": ["calm", "sharp"],
                            "sliders": [],
                        }
                    ],
                }
            ),
            source="llamacpp",
        )


def main() -> None:
    original_mode = conversation.CONVERSATION_MODE
    conversation.CONVERSATION_MODE = "model_assisted"
    try:
        profile = conversation.new_profile()
        profile.ideas.append("I want something about patience.")
        profile.free_notes.append("quiet pressure")
        profile.free_notes.append("hidden order")
        profile.visual_preferences.update({"minimal_rich": 35, "calm_intense": 45, "geometric_organic": 45})
        profile = conversation.update_scores(profile)
        fallback = conversation.choose_conversation_turn(profile)

        good = conversation.choose_conversation_turn_with_model(profile, fallback, GoodClient())
        assert good.controls[0].kind == ControlKind.SWATCH_PICKER
        assert good.controls[0].options[0] == "ash, gold, pale blue"

        repaired = conversation.choose_conversation_turn_with_model(profile, fallback, RepairClient())
        assert repaired.controls[0].kind == ControlKind.SWATCH_PICKER
        assert repaired.controls[0].options[0] == "rain glass, graphite, low amber"

        wrong_stage = conversation.choose_conversation_turn_with_model(profile, fallback, WrongStageClient())
        assert wrong_stage.controls[0].kind == ControlKind.TEXT_REFINEMENT
        assert wrong_stage.progress_label == "Model: retry needed"

        bad = conversation.choose_conversation_turn_with_model(profile, fallback, BadClient())
        assert bad.controls[0].kind == ControlKind.TEXT_REFINEMENT
        assert bad.progress_label == "Model: retry needed"
    finally:
        conversation.CONVERSATION_MODE = original_mode

    print("model_assisted_ok")


if __name__ == "__main__":
    main()
