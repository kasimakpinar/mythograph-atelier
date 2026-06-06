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


class ControlKind(StrEnum):
    CHOICE_CARDS = "choice_cards"
    MULTI_CHOICE_CARDS = "multi_choice_cards"
    SLIDER_GROUP = "slider_group"
    SWATCH_PICKER = "swatch_picker"
    TEXT_REFINEMENT = "text_refinement"
    READY_BUTTON = "ready_button"


class SliderSpec(BaseModel):
    key: str
    label: str
    left_label: str
    right_label: str
    value: int = 50


class DynamicControl(BaseModel):
    kind: ControlKind
    label: str
    prompt: str
    options: list[str] = Field(default_factory=list)
    sliders: list[SliderSpec] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    assistant_message: str
    progress_label: str
    reason: str
    is_ready: bool = False
    controls: list[DynamicControl] = Field(default_factory=list)


class ControlResponse(BaseModel):
    kind: ControlKind
    values: list[str] = Field(default_factory=list)
    text: str = ""
    sliders: dict[str, float] = Field(default_factory=dict)
