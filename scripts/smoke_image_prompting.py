from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.services.image_prompting import build_flux_prompt


def main() -> None:
    prompt = "A quiet abstract painting about relief, no text."
    assert build_flux_prompt(
        prompt,
        use_lora=True,
        prepend_trigger=True,
        trigger="MYTHABS1",
    ) == "MYTHABS1. A quiet abstract painting about relief, no text."
    assert build_flux_prompt(
        "MYTHABS1. MYTHABS1. A quiet abstract painting about relief, no text.",
        use_lora=True,
        prepend_trigger=True,
        trigger="MYTHABS1",
    ) == "MYTHABS1. A quiet abstract painting about relief, no text."
    assert build_flux_prompt(
        "MYTHABS1. A quiet abstract painting about relief, no text.",
        use_lora=False,
        trigger="MYTHABS1",
    ) == "A quiet abstract painting about relief, no text."
    assert build_flux_prompt(
        prompt,
        use_lora=True,
        prepend_trigger=False,
        trigger="MYTHABS1",
    ) == "A quiet abstract painting about relief, no text."
    print("image_prompting_ok")


if __name__ == "__main__":
    main()
