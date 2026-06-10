import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.models.llm_client import LLMResponse
from mythograph.services.starters import generate_conversation_starters


class StarterClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None, thinking=False):
        assert user_payload["task"] == "conversation_starters"
        assert response_format == {"type": "json_object"}
        assert thinking is False
        return LLMResponse(
            content=json.dumps(
                {
                    "starters": [
                        {"title": "Small relief", "text": "I want something about finally breathing easier."},
                        {"title": "Spring joy", "text": "The joy of spring after a long winter."},
                        {"title": "Missing home", "text": "I miss home, but I am also proud that I left."},
                        {"title": "After a hard week", "text": "I want it to feel like a clean room after a hard week."},
                    ]
                }
            ),
            source="llamacpp",
        )


def main() -> None:
    starters = generate_conversation_starters(StarterClient(), count=4)
    assert len(starters) == 4
    assert starters[0].title == "Small relief"
    assert "breathing easier" in starters[0].text
    print("conversation_starters_ok")


if __name__ == "__main__":
    main()
