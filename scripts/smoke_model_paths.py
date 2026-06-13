import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.models.llm_client import LLMResponse
from mythograph.services.art_recipe import _recipe_token_budget, build_art_recipe_with_model
from mythograph.services.interview import new_profile


class FakeClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None, thinking=False):
        assert max_tokens == _recipe_token_budget()
        assert response_format == {"type": "json_object"}
        assert thinking is True
        assert user_payload["task"] == "final_art_recipe"
        assert user_payload["thinking_mode"] is True
        assert "connection_principle" in user_payload
        return LLMResponse(
            content=json.dumps(
                {
                    "title": "Test Title",
                    "central_phrase": "A quiet line can still know where it is going.",
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
                        "The line carries the feeling of choosing direction under pressure. "
                        "The open threshold gives that direction room to change without losing itself."
                    ),
                }
            ),
            source="local",
        )


class PartialRecipeClient:
    def complete_json(self, system_prompt, user_payload, max_tokens=None, temperature=None, response_format=None, thinking=False):
        assert thinking is True
        return LLMResponse(
            content=json.dumps(
                {
                    "title": "Bright Afterlight",
                    "central_phrase": "Joy can arrive softly after the heavy part has passed.",
                    "main_idea": "joy arriving after a heavy mood",
                    "visual_style": "soft, organic, hushed",
                    "palette": ["#dfe8ee", "#f6d86b", "#263238"],
                    "symbols": [
                        {"visual": "soft radial glow", "meaning": "arrival as silent joy"},
                        {"visual": "wide quiet color plane", "meaning": "room after heaviness"},
                        {"visual": "small warm edge marks", "meaning": "joy beginning to move"},
                    ],
                    "composition": "soft radial movement crossing a cleared field",
                    "image_prompt": "Abstract painting, soft radial glow, cleared field, no text",
                    "negative_prompt": "text, letters, watermark",
                    "friend_explanation": "A small glow shows joy after heaviness.",
                }
            ),
            source="llamacpp",
        )


def main() -> None:
    profile = new_profile()
    profile.ideas.append("I want something about the peace in loneliness")
    recipe = build_art_recipe_with_model(profile, client=FakeClient())
    assert "is not decoration" not in recipe.friend_explanation.lower()
    assert recipe.central_phrase

    bright_profile = new_profile()
    bright_profile.ideas.extend(["I want something about joy arriving after a heavy mood", "glow", "splash"])
    bright_profile.free_notes.append("pulse")
    bright_profile.visual_preferences["palette_mood"] = "hushed"
    bright_recipe = build_art_recipe_with_model(bright_profile, client=PartialRecipeClient())
    assert len(bright_recipe.symbols) >= 3
    assert "is not decoration" not in bright_recipe.friend_explanation.lower()
    assert bright_recipe.central_phrase
    print(recipe.title)


if __name__ == "__main__":
    main()
