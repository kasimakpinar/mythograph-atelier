import time

from mythograph.config import IMAGE_MODE
from mythograph.config import IMAGE_DTYPE, IMAGE_HEIGHT, IMAGE_MODEL_ID, IMAGE_SEED, IMAGE_STEPS, IMAGE_WIDTH
from mythograph.config import OUTPUT_DIR
from mythograph.schemas.art_recipe import ArtRecipe
from mythograph.services.generation import ImageGenerationResult, generate_fallback_image


class ImageClient:
    _flux_pipe = None

    def __init__(self, mode: str = IMAGE_MODE) -> None:
        self.mode = mode

    def generate(self, recipe: ArtRecipe) -> ImageGenerationResult:
        if self.mode in {"pillow", "mock", "fallback"}:
            return generate_fallback_image(recipe)
        if self.mode in {"flux", "flux_klein", "diffusers"}:
            return self._generate_flux(recipe)

        fallback = generate_fallback_image(recipe)
        fallback.source = "pillow_fallback"
        fallback.error = f"Image mode '{self.mode}' is not implemented yet."
        return fallback

    def _generate_flux(self, recipe: ArtRecipe) -> ImageGenerationResult:
        started = time.perf_counter()
        try:
            pipe = self._load_flux_pipeline()
            image_kwargs = {
                "prompt": recipe.image_prompt,
                "num_inference_steps": IMAGE_STEPS,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
            }
            negative_prompt = recipe.negative_prompt
            try:
                import torch

                image_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(IMAGE_SEED)
            except Exception:
                pass

            image = self._run_flux_pipeline(pipe, image_kwargs, negative_prompt).images[0]
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUTPUT_DIR / f"flux_{abs(hash(recipe.image_prompt)) % 10_000_000}.png"
            image.save(path)
            return ImageGenerationResult(
                path=str(path),
                source="flux_klein",
                elapsed_seconds=round(time.perf_counter() - started, 3),
            )
        except Exception as exc:
            fallback = generate_fallback_image(recipe)
            fallback.error = f"FLUX backend failed: {exc}"
            return fallback

    @classmethod
    def _load_flux_pipeline(cls):
        if cls._flux_pipe is not None:
            return cls._flux_pipe

        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError("Install requirements-image.txt to enable MYTHOGRAPH_IMAGE_MODE=flux.") from exc

        cls._flux_pipe = DiffusionPipeline.from_pretrained(
            IMAGE_MODEL_ID,
            dtype=_torch_dtype(torch),
            device_map="cuda",
        )
        return cls._flux_pipe

    @staticmethod
    def _run_flux_pipeline(pipe, image_kwargs, negative_prompt):
        try:
            return pipe(**image_kwargs, negative_prompt=negative_prompt)
        except TypeError as exc:
            if "negative_prompt" not in str(exc):
                raise
            return pipe(**image_kwargs)


def _torch_dtype(torch):
    if IMAGE_DTYPE in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if IMAGE_DTYPE in {"fp32", "float32"}:
        return torch.float32
    return torch.float16
