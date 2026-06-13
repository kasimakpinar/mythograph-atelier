from enum import StrEnum

from pydantic import BaseModel, Field


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
    label: str = ""
    prompt: str = ""


class ConversationStarter(BaseModel):
    title: str
    text: str


class ConversationStarters(BaseModel):
    starters: list[ConversationStarter] = Field(default_factory=list)
