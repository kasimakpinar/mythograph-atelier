import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.models.llm_client import LLMResponse
from mythograph.config import LLM_RECIPE_MAX_TOKENS
from mythograph.services.art_recipe import build_art_recipe_with_model
from mythograph.services.interview import choose_next_ui_with_model, new_profile


class FakeClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None):
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
        assert response_format == {"type": "json_object"}
        assert "fallback_recipe" not in user_payload
        assert "connection_principle" in user_payload
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
                    "friend_explanation": (
                        "A line is not decoration; it is inner direction. "
                        "A door gives it a second force: a threshold into change."
                    ),
                }
            ),
            source="local",
        )


class PartialRecipeClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None):
        return LLMResponse(
            content=json.dumps(
                {
                    "title": "Rain Afterlight",
                    "main_idea": "joy brought by the rain",
                    "visual_style": "soft, organic, hushed",
                    "palette": ["#dfe8ee", "#f6d86b", "#263238"],
                    "symbols": [{"visual": "soft radial glow", "meaning": "arrival as silent joy"}],
                    "composition": "soft radial movement crossing a rain-washed field",
                    "image_prompt": "Abstract painting, soft radial glow, rain-washed field, no text",
                    "negative_prompt": "text, letters, watermark",
                    "friend_explanation": "A small glow shows joy after rain.",
                }
            ),
            source="llamacpp",
        )


def main() -> None:
    profile = new_profile()
    profile.ideas.append("I want something about the peace in loneliness")
    next_ui = choose_next_ui_with_model(profile, FakeClient())
    recipe = build_art_recipe_with_model(profile, client=FakeClient())
    assert "is not decoration" not in recipe.friend_explanation.lower()

    rain_profile = new_profile()
    rain_profile.ideas.extend(["I want something about the joy brought by the rain", "glow", "splash"])
    rain_profile.free_notes.append("pulse")
    rain_profile.visual_preferences["palette_mood"] = "hushed"
    rain_recipe = build_art_recipe_with_model(rain_profile, client=PartialRecipeClient())
    assert len(rain_recipe.symbols) >= 3
    assert "rain" in rain_recipe.friend_explanation.lower()
    assert "is not decoration" not in rain_recipe.friend_explanation.lower()
    print(next_ui.next_action)
    print(recipe.title)


if __name__ == "__main__":
    main()
