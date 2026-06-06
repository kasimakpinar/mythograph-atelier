import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.models.llm_client import LLMClient


class FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}]}).encode("utf-8")


def main() -> None:
    captured = {}
    original_urlopen = urllib.request.urlopen

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["url"] = request.full_url
        return FakeHTTPResponse()

    urllib.request.urlopen = fake_urlopen
    try:
        client = LLMClient(
            mode="local",
            base_url="https://example.test/v1",
            model="test-model",
            timeout=12,
            api_key="secret-token",
            source_label="modal",
            max_tokens=321,
            temperature=0.2,
        )
        response = client.complete_json("system", {"hello": "world"})
    finally:
        urllib.request.urlopen = original_urlopen

    assert response.source == "modal"
    assert response.content == "{\"ok\": true}"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    print("llm_auth_ok")


if __name__ == "__main__":
    main()
