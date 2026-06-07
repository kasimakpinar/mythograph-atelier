from pydantic import BaseModel, Field


class ObjectiveScores(BaseModel):
    idea_anchor: float = 0.0
    visual_taste: float = 0.0
    symbolic_material: float = 0.0
    surprise_level: float = 0.0
    ready_to_generate: bool = False


class InterviewProfile(BaseModel):
    ideas: list[str] = Field(default_factory=list)
    visual_preferences: dict[str, int | float | str] = Field(default_factory=dict)
    styles: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    contrasts: list[str] = Field(default_factory=list)
    free_notes: list[str] = Field(default_factory=list)
    asked_questions: list[str] = Field(default_factory=list)
    offered_options: list[str] = Field(default_factory=list)
    turn_count: int = 0
    scores: ObjectiveScores = Field(default_factory=ObjectiveScores)
