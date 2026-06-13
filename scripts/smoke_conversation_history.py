import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.models.llm_client import LLMResponse
from mythograph.schemas.ui import ControlKind
import mythograph.services.conversation as conversation


class HistoryClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None, thinking=False):
        assert user_payload["conversation_history"]
        assert user_payload["conversation_history"][-1]["role"] == "user"
        return LLMResponse(
            content=json.dumps(
                {
                    "assistant_message": "That gives me the emotional center. I can turn this into the artwork now.",
                    "progress_label": "Ready",
                    "reason": "the conversation has enough personal signal",
                    "is_ready": True,
                    "controls": [
                        {
                            "kind": "ready_button",
                            "label": "Create artwork",
                            "prompt": "Generate the painting",
                            "options": ["Create artwork"],
                            "sliders": [],
                        }
                    ],
                }
            ),
            source="llamacpp",
        )


def main() -> None:
    profile = conversation.new_profile()
    profile.ideas.append("I want something about caring and guilt.")
    profile.free_notes.extend(["Both can be true.", "Resistance matters here."])
    profile.turn_count = 3
    profile = conversation.update_scores(profile)
    profile, turn = conversation.advance_conversation(
        profile,
        user_message="Resistance matters here.",
        client=HistoryClient(),
        conversation_history=[
            {"role": "user", "content": "I want something about caring and guilt."},
            {"role": "assistant", "content": "What part feels central?"},
            {"role": "user", "content": "Both can be true. Resistance matters here."},
        ],
    )
    assert turn.is_ready
    assert turn.controls[0].kind == ControlKind.READY_BUTTON
    print("conversation_history_ok")


if __name__ == "__main__":
    main()
