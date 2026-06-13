import time

from mythograph.config import IMAGE_MODE
from mythograph.config import (
    FT_IMAGE_MODEL_PATH,
    FT_IMAGE_MODEL_REPO,
    FT_IMAGE_MODEL_SCALE,
    FT_IMAGE_MODEL_TRIGGER,
    FT_IMAGE_MODEL_WEIGHT_NAME,
    IMAGE_CPU_OFFLOAD,
    IMAGE_DTYPE,
    IMAGE_GUIDANCE_SCALE,
    IMAGE_HEIGHT,
    IMAGE_MODEL_ID,
    IMAGE_SEED,
    IMAGE_STEPS,
    IMAGE_WIDTH,
    USE_FT_IMAGE_MODEL,
    FT_IMAGE_MODEL_PREPEND_TRIGGER,
)
from mythograph.config import OUTPUT_DIR
from mythograph.schemas.art_recipe import ArtRecipe
from mythograph.services.generation import ImageGenerationResult, generate_fallback_image
from mythograph.services.image_prompting import build_flux_prompt
from mythograph.services.trace_logger import log_event


class ImageClient:
    _flux_pipe = None
    _flux_pipe_key = None

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
            internal_prompt = build_flux_prompt(
                recipe.image_prompt,
                use_lora=USE_FT_IMAGE_MODEL,
                prepend_trigger=FT_IMAGE_MODEL_PREPEND_TRIGGER,
                trigger=FT_IMAGE_MODEL_TRIGGER,
            )
            log_event(
                "flux_generation_config",
                {
                    "use_ft_image_model": USE_FT_IMAGE_MODEL,
                    "prepend_trigger": FT_IMAGE_MODEL_PREPEND_TRIGGER,
                    "trigger": FT_IMAGE_MODEL_TRIGGER,
                    "ft_image_model_repo": FT_IMAGE_MODEL_REPO,
                    "ft_image_model_path": FT_IMAGE_MODEL_PATH,
                    "ft_image_model_weight_name": FT_IMAGE_MODEL_WEIGHT_NAME,
                    "ft_image_model_scale": FT_IMAGE_MODEL_SCALE,
                    "model_id": IMAGE_MODEL_ID,
                },
            )
            image_kwargs = {
                "prompt": internal_prompt,
                "guidance_scale": IMAGE_GUIDANCE_SCALE,
                "num_inference_steps": IMAGE_STEPS,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
            }
            negative_prompt = recipe.negative_prompt
            try:
                import torch

                if IMAGE_SEED > 0:
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
        pipe_key = cls._flux_pipeline_key()
        if cls._flux_pipe is not None and cls._flux_pipe_key == pipe_key:
            return cls._flux_pipe

        try:
            import torch
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise RuntimeError("Install requirements-image.txt to enable MYTHOGRAPH_IMAGE_MODE=flux.") from exc

        cls._flux_pipe = Flux2KleinPipeline.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=_torch_dtype(torch),
        )
        if USE_FT_IMAGE_MODEL:
            lora_source = FT_IMAGE_MODEL_PATH or FT_IMAGE_MODEL_REPO
            weight_name = FT_IMAGE_MODEL_WEIGHT_NAME or None
            cls._flux_pipe.load_lora_weights(lora_source, weight_name=weight_name)
            if hasattr(cls._flux_pipe, "set_adapters"):
                try:
                    cls._flux_pipe.set_adapters(["default"], adapter_weights=[FT_IMAGE_MODEL_SCALE])
                except Exception:
                    pass
        if IMAGE_CPU_OFFLOAD:
            cls._flux_pipe.enable_model_cpu_offload()
        else:
            cls._flux_pipe.to("cuda")
        cls._flux_pipe_key = pipe_key
        return cls._flux_pipe

    @staticmethod
    def _flux_pipeline_key():
        return (
            IMAGE_MODEL_ID,
            IMAGE_DTYPE,
            IMAGE_CPU_OFFLOAD,
            USE_FT_IMAGE_MODEL,
            FT_IMAGE_MODEL_PATH,
            FT_IMAGE_MODEL_REPO,
            FT_IMAGE_MODEL_WEIGHT_NAME,
            FT_IMAGE_MODEL_SCALE,
        )

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
