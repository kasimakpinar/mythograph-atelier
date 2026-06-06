from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.config import IMAGE_HEIGHT, IMAGE_MODE, IMAGE_MODEL_ID, IMAGE_STEPS, IMAGE_WIDTH


def main() -> None:
    print(IMAGE_MODE)
    print(IMAGE_MODEL_ID)
    print(f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(IMAGE_STEPS)


if __name__ == "__main__":
    main()
