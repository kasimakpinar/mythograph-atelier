import json
import os
import site
import sys
import ctypes
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mythograph.config import (
    LLAMACPP_CHAT_FORMAT,
    LLAMACPP_FILENAME,
    LLAMACPP_N_CTX,
    LLAMACPP_N_GPU_LAYERS,
    LLAMACPP_REPO_ID,
    LLAMACPP_VERBOSE,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODE,
    LLM_MODEL,
    LLM_SOURCE_LABEL,
    LLM_TEMPERATURE,
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
        api_key: str = LLM_API_KEY,
        source_label: str = LLM_SOURCE_LABEL,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> None:
        self.mode = mode
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.source_label = source_label
        self.max_tokens = max_tokens
        self.temperature = temperature

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
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            source = self.source_label or "local"
            return LLMResponse(content=content, source=source, raw=raw)
        except (KeyError, json.JSONDecodeError, TimeoutError, urllib.error.URLError, OSError) as exc:
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
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
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
            _preload_cuda_libraries()
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
        "openai_compatible_source_label": LLM_SOURCE_LABEL or "local",
        "openai_compatible_auth": bool(LLM_API_KEY),
        "llm_max_tokens": LLM_MAX_TOKENS,
        "llm_temperature": LLM_TEMPERATURE,
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


def _preload_cuda_libraries() -> None:
    if not sys.platform.startswith("linux"):
        return

    library_dirs = _cuda_library_dirs()
    if not library_dirs:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    additions = [str(path) for path in library_dirs if str(path) not in existing.split(":")]
    if additions:
        os.environ["LD_LIBRARY_PATH"] = ":".join(additions + ([existing] if existing else []))

    for library_name in (
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
    ):
        for directory in library_dirs:
            library_path = directory / library_name
            if library_path.exists():
                try:
                    ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


def _cuda_library_dirs() -> list[Path]:
    roots = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))

    library_dirs: list[Path] = []
    for root in roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        for path in nvidia_root.glob("*/lib"):
            if path.is_dir():
                library_dirs.append(path)
    return library_dirs


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
