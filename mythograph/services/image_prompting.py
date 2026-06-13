import os
import re
from typing import Optional


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_bool_any(primary: str, secondary: str, default: bool = False) -> bool:
    if os.getenv(primary) is not None:
        return env_bool(primary, default)
    return env_bool(secondary, default)


def normalize_lora_trigger(trigger: str) -> str:
    return trigger.strip().strip(".:;,- ")


def strip_existing_trigger(prompt: str, trigger: str) -> str:
    trigger = normalize_lora_trigger(trigger)
    if not trigger:
        return prompt.strip()

    pattern = rf"^(\s*{re.escape(trigger)}\s*[\.\:\-\u2013\,]?\s*)+"
    return re.sub(pattern, "", prompt.strip(), flags=re.IGNORECASE).strip()


def build_flux_prompt(
    image_prompt: str,
    *,
    use_lora: Optional[bool] = None,
    prepend_trigger: Optional[bool] = None,
    trigger: Optional[str] = None,
) -> str:
    if use_lora is None:
        use_lora = env_bool_any("USE_FT_IMAGE_MODEL", "USE_MYTHOGRAPH_LORA", default=True)

    if prepend_trigger is None:
        prepend_trigger = env_bool_any("FT_IMAGE_MODEL_PREPEND_TRIGGER", "MYTHOGRAPH_LORA_PREPEND_TRIGGER", default=True)

    trigger = normalize_lora_trigger(
        trigger
        or os.getenv("FT_IMAGE_MODEL_TRIGGER")
        or os.getenv("MYTHOGRAPH_LORA_TRIGGER")
        or "MYTHABS1"
    )

    prompt = (image_prompt or "").strip()

    if not use_lora or not prepend_trigger or not trigger:
        return strip_existing_trigger(prompt, trigger)

    prompt_without_trigger = strip_existing_trigger(prompt, trigger)
    return f"{trigger}. {prompt_without_trigger}"
