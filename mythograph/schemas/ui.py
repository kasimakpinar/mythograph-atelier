from enum import StrEnum

from pydantic import BaseModel, Field


class UIAction(StrEnum):
    ASK_FREE_TEXT = "ask_free_text"
    SHOW_IDEA_CARDS = "show_idea_cards"
    SHOW_VISUAL_SLIDERS = "show_visual_sliders"
    SHOW_STYLE_CARDS = "show_style_cards"
    SHOW_SYMBOL_CARDS = "show_symbol_cards"
    ASK_CONTRAST = "ask_contrast"
    SURPRISE_STEP = "surprise_step"
    READY_TO_GENERATE = "ready_to_generate"


class NextUI(BaseModel):
    assistant_message: str
    next_action: UIAction
    reason: str
    question: str
    options: list[str] = Field(default_factory=list)
