import json
from typing import Any

from mythograph.config import DATA_DIR, TRACE_PATH
from mythograph.schemas.trace import TraceEvent


def log_event(event_type: str, payload: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    event = TraceEvent(event_type=event_type, payload=payload)
    with TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.model_dump(), ensure_ascii=True) + "\n")


def export_trace() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TRACE_PATH.exists():
        log_event("trace_export", {"message": "No prior trace events were found."})

    export_path = DATA_DIR / "mythograph_trace_export.jsonl"
    lines: list[str] = []
    with TRACE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            lines.append(json.dumps(_redact_event(event), ensure_ascii=True))
    export_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(export_path)


def _redact_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if isinstance(payload, dict):
        payload = _redact_value(payload)
    return {**event, "payload": payload}


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"raw_content"}:
                redacted[key] = "[redacted from public trace export]"
            else:
                redacted[key] = _redact_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value
