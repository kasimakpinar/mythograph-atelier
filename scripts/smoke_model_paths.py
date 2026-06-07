import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.models.llm_client import LLMResponse
from mythograph.config import LLM_RECIPE_MAX_TOKENS
from mythograph.services.art_recipe import build_art_recipe_with_model
from mythograph.services.interview import choose_next_ui_with_model, new_profile


class FakeClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None):
        if "Choose the next UI action" in system_prompt:
            return LLMResponse(
                content=json.dumps(
                    {
                        "assistant_message": "Test message",
                        "next_action": "show_style_cards",
                        "reason": "fake model path",
                        "question": "Pick a style",
                        "options": ["quiet and elegant", "bold and dramatic"],
                    }
                ),
                source="local",
            )
        assert max_tokens == LLM_RECIPE_MAX_TOKENS
        return LLMResponse(
            content=json.dumps(
                {
                    "title": "Test Title",
                    "main_idea": "Test idea",
                    "visual_style": "minimal geometric",
                    "palette": ["#111111", "#eeeeee", "#b8872f"],
                    "symbols": [
                        {"visual": "a line", "meaning": "direction"},
                        {"visual": "a door", "meaning": "change"},
                        {"visual": "open space", "meaning": "freedom"},
                    ],
                    "composition": "balanced abstract composition",
                    "image_prompt": "Abstract painting, no text, no watermark",
                    "negative_prompt": "text, letters, watermark",
                    "friend_explanation": "A line carries direction through change.",
                }
            ),
            source="local",
        )


def main() -> None:
    profile = new_profile()
    next_ui = choose_next_ui_with_model(profile, FakeClient())
    recipe = build_art_recipe_with_model(profile, client=FakeClient())
    print(next_ui.next_action)
    print(recipe.title)


if __name__ == "__main__":
    main()
