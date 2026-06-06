import json
from typing import Any

from mythograph.config import DATA_DIR, TRACE_PATH
from mythograph.schemas.trace import TraceEvent


def log_event(event_type: str, payload: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    event = TraceEvent(event_type=event_type, payload=payload)
    with TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.model_dump(), ensure_ascii=True) + "\n")
