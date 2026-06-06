from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.config import IMAGE_DTYPE, IMAGE_HEIGHT, IMAGE_MODE, IMAGE_MODEL_ID, IMAGE_STEPS, IMAGE_WIDTH
from mythograph.models.image_client import ImageClient


class FakePipelineResult:
    images = ["ok"]


class FakeNoNegativePromptPipeline:
    def __call__(self, **kwargs):
        if "negative_prompt" in kwargs:
            raise TypeError("unexpected keyword argument negative_prompt")
        return FakePipelineResult()


def main() -> None:
    print(IMAGE_MODE)
    print(IMAGE_MODEL_ID)
    print(f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(IMAGE_STEPS)
    print(IMAGE_DTYPE)
    print(ImageClient._run_flux_pipeline(FakeNoNegativePromptPipeline(), {"prompt": "x"}, "avoid text").images[0])


if __name__ == "__main__":
    main()
