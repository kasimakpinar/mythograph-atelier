import time

from mythograph.config import LLM_CHAT_MAX_TOKENS, ROOT_DIR
from mythograph.models.llm_client import LLMClient, extract_json_object
from mythograph.schemas.ui import ConversationStarter, ConversationStarters
from mythograph.services.trace_logger import log_event
from mythograph.ui.examples import STARTER_IDEAS


FALLBACK_STARTERS = [
    ConversationStarter(title=title, text=text)
    for title, text in [
        ("A feeling", "I want something about being tired but still trying."),
        ("A season", "The joy of spring after a long winter."),
        ("A memory", "I miss a place that does not exist anymore."),
        ("A contradiction", "I feel free, but also strangely lonely."),
        ("After a hard week", "I want it to feel like a clean room after a hard week."),
        ("A strange sentence", "I am not lost, but I do not recognize the road."),
    ]
]


def fallback_starters(count: int = 6) -> list[ConversationStarter]:
    starters = FALLBACK_STARTERS or [ConversationStarter(title=title, text=text) for title, text in STARTER_IDEAS]
    return starters[:count]


def generate_conversation_starters(client: LLMClient | None = None, count: int = 6) -> list[ConversationStarter]:
    llm = client or LLMClient()
    system_prompt = (ROOT_DIR / "mythograph" / "prompts" / "conversation_director_system.txt").read_text(
        encoding="utf-8"
    )
    payload = {
        "task": "conversation_starters",
        "count": count,
        "tone": "casual, warm, varied, understandable",
        "goal": "Generate ordinary human starting examples for Mythograph Atelier.",
        "avoid": ["heavy art jargon", "generic inspirational quotes", "overly poetic abstractions"],
    }
    started = time.perf_counter()
    response = llm.complete_json(
        system_prompt,
        payload,
        max_tokens=max(LLM_CHAT_MAX_TOKENS, 220),
        response_format={"type": "json_object"},
    )
    if response.source == "mock":
        return fallback_starters(count)

    current_response = response
    last_error = response.error or ""
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            if current_response.error:
                raise ValueError(current_response.error)
            starters = _normalize_starters(extract_json_object(current_response.content), count)
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_attempts - 1:
                starters = fallback_starters(count)
                log_event(
                    "llm_conversation_starters",
                    {
                        "source": current_response.source or response.source,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "error": last_error,
                        "transport_error": current_response.error,
                        "used_fallback": True,
                        "retry_count": attempt,
                        "raw_content": current_response.content or response.content,
                        "starters": [starter.model_dump() for starter in starters],
                    },
                )
                return starters
            current_response = llm.complete_json(
                _repair_starters_prompt(),
                {
                    "invalid_json": current_response.content,
                    "previous_error": last_error,
                    "attempt": attempt + 1,
                    "required_shape": {"starters": [{"title": "string", "text": "string"}]},
                    "fallback_examples": [starter.model_dump() for starter in fallback_starters(count)],
                },
                max_tokens=max(LLM_CHAT_MAX_TOKENS, 220),
                response_format={"type": "json_object"},
            )
            continue

        log_event(
            "llm_conversation_starters",
            {
                "source": current_response.source,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "transport_error": current_response.error,
                "used_fallback": False,
                "retry_count": attempt,
                "raw_content": current_response.content,
                "starters": [starter.model_dump() for starter in starters],
            },
        )
        return starters
    return fallback_starters(count)


def _repair_starters_prompt() -> str:
    return (
        "Return valid JSON only. Shape: "
        '{"starters":[{"title":"string","text":"string"}]}. '
        "Make starters ordinary, human, casual, and easy to click."
    )


def _normalize_starters(payload: dict, count: int) -> list[ConversationStarter]:
    parsed = ConversationStarters.model_validate(payload)
    starters: list[ConversationStarter] = []
    seen: set[str] = set()
    for starter in parsed.starters:
        title = _clean_text(starter.title, 34)
        text = _clean_text(starter.text, 140)
        key = text.lower()
        if not title or not text or key in seen:
            continue
        if _sounds_overwritten(title) or _sounds_overwritten(text):
            continue
        seen.add(key)
        starters.append(ConversationStarter(title=title, text=text))
        if len(starters) >= count:
            break
    for starter in fallback_starters(count):
        if len(starters) >= count:
            break
        if starter.text.lower() not in seen:
            starters.append(starter)
    return starters[:count]


def _clean_text(value: str, limit: int) -> str:
    clean = " ".join(str(value or "").strip().split())
    if len(clean) > limit:
        clean = clean[: limit - 1].rstrip(" ,.;:") + "."
    return clean


def _sounds_overwritten(text: str) -> bool:
    lowered = text.lower()
    art_words = ["luminous", "ancestral", "architecture", "trembling", "forgotten tenderness"]
    return any(word in lowered for word in art_words)
