import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.models.llm_client import LLMResponse
import mythograph.services.conversation as conversation


class GoodClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None, thinking=False):
        assert response_format == {"type": "json_object"}
        assert "conversation_history" in user_payload
        assert "atelier_state" in user_payload
        assert set(user_payload).issuperset({"task", "conversation_history", "atelier_state", "can_generate"})
        return LLMResponse(
            content=json.dumps(
                {
                    "assistant_message": "That sounds like care under pressure. What should the painting understand about that pressure?",
                    "progress_label": "Listening",
                    "reason": "the model chose a natural follow-up",
                    "is_ready": False,
                    "controls": [],
                }
            ),
            source="llamacpp",
        )


class RepairClient:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None, thinking=False):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="not json", source="llamacpp")
        return LLMResponse(
            content=json.dumps(
                {
                    "assistant_message": "I can use that. What does that pressure ask you to protect?",
                    "progress_label": "Listening",
                    "reason": "repair returned valid JSON",
                    "is_ready": False,
                    "controls": [],
                }
            ),
            source="llamacpp",
        )


class BadClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None, thinking=False):
        return LLMResponse(content="not json", source="llamacpp")


def _profile():
    profile = conversation.new_profile()
    profile.ideas.append("I want something about caring and guilt.")
    profile.free_notes.append("Both can be true.")
    profile.turn_count = 2
    return conversation.update_scores(profile)


def main() -> None:
    history = [
        {"role": "user", "content": "I want something about caring and guilt."},
        {"role": "assistant", "content": "Tell me more."},
        {"role": "user", "content": "Both can be true."},
    ]

    good = conversation.choose_conversation_turn_with_model(_profile(), GoodClient(), history)
    assert good.controls == []
    assert "care under pressure" in good.assistant_message

    repair_client = RepairClient()
    repaired = conversation.choose_conversation_turn_with_model(_profile(), repair_client, history)
    assert repair_client.calls == 2
    assert repaired.controls == []

    bad = conversation.choose_conversation_turn_with_model(_profile(), BadClient(), history)
    assert bad.controls == []
    assert not bad.is_ready
    assert "could not produce a valid next step" in bad.assistant_message

    print("model_assisted_ok")


if __name__ == "__main__":
    main()
