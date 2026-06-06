from mythograph.config import IMAGE_MODE
from mythograph.schemas.art_recipe import ArtRecipe
from mythograph.services.generation import ImageGenerationResult, generate_fallback_image


class ImageClient:
    def __init__(self, mode: str = IMAGE_MODE) -> None:
        self.mode = mode

    def generate(self, recipe: ArtRecipe) -> ImageGenerationResult:
        if self.mode in {"pillow", "mock", "fallback"}:
            return generate_fallback_image(recipe)

        fallback = generate_fallback_image(recipe)
        fallback.source = "pillow_fallback"
        fallback.error = f"Image mode '{self.mode}' is not implemented yet."
        return fallback
