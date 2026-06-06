import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from mythograph.config import (
    LLAMACPP_CHAT_FORMAT,
    LLAMACPP_FILENAME,
    LLAMACPP_N_CTX,
    LLAMACPP_N_GPU_LAYERS,
    LLAMACPP_REPO_ID,
    LLAMACPP_VERBOSE,
    LLM_BASE_URL,
    LLM_MODE,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)


@dataclass
class LLMResponse:
    content: str
    source: str
    raw: dict[str, Any] | None = None
    error: str | None = None


class LLMClient:
    def __init__(
        self,
        mode: str = LLM_MODE,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        timeout: float = LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.mode = mode
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def complete_json(self, system_prompt: str, user_payload: dict[str, Any]) -> LLMResponse:
        if self.mode == "mock":
            return LLMResponse(content="", source="mock")
        if self.mode == "llamacpp":
            return LlamaCppClient().complete_json(system_prompt, user_payload)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
            "temperature": 0.7,
            "max_tokens": 1200,
        }

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            return LLMResponse(content=content, source="local", raw=raw)
        except (KeyError, json.JSONDecodeError, TimeoutError, urllib.error.URLError) as exc:
            return LLMResponse(content="", source="fallback", error=str(exc))


class LlamaCppClient:
    _llm: Any = None

    def complete_json(self, system_prompt: str, user_payload: dict[str, Any]) -> LLMResponse:
        try:
            llm = self._load()
            raw = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
                ],
                temperature=0.7,
                max_tokens=1200,
            )
            content = raw["choices"][0]["message"]["content"]
            return LLMResponse(content=content, source="llamacpp", raw=raw)
        except Exception as exc:
            return LLMResponse(content="", source="fallback", error=f"llama.cpp unavailable: {exc}")

    @classmethod
    def _load(cls) -> Any:
        if cls._llm is not None:
            return cls._llm

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "Install llama-cpp-python and huggingface-hub, or use MYTHOGRAPH_LLM_MODE=mock/local."
            ) from exc

        kwargs: dict[str, Any] = {
            "repo_id": LLAMACPP_REPO_ID,
            "filename": LLAMACPP_FILENAME,
            "n_ctx": LLAMACPP_N_CTX,
            "n_gpu_layers": LLAMACPP_N_GPU_LAYERS,
            "verbose": LLAMACPP_VERBOSE,
        }
        if LLAMACPP_CHAT_FORMAT:
            kwargs["chat_format"] = LLAMACPP_CHAT_FORMAT

        cls._llm = Llama.from_pretrained(**kwargs)
        return cls._llm


def runtime_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "mode": LLM_MODE,
        "openai_compatible_base_url": LLM_BASE_URL,
        "openai_compatible_model": LLM_MODEL,
    }
    if LLM_MODE == "llamacpp":
        status.update(
            {
                "llamacpp_repo_id": LLAMACPP_REPO_ID,
                "llamacpp_filename": LLAMACPP_FILENAME,
                "llamacpp_n_ctx": LLAMACPP_N_CTX,
                "llamacpp_n_gpu_layers": LLAMACPP_N_GPU_LAYERS,
            }
        )
    return status


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(stripped[start : end + 1])
