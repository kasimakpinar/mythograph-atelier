from pydantic import BaseModel, Field


class Symbol(BaseModel):
    visual: str
    meaning: str


class ArtRecipe(BaseModel):
    title: str
    main_idea: str
    visual_style: str
    palette: list[str] = Field(min_length=3, max_length=6)
    symbols: list[Symbol] = Field(min_length=3, max_length=6)
    composition: str
    image_prompt: str
    negative_prompt: str
    friend_explanation: str
