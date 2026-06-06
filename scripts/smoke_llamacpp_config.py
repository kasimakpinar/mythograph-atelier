from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mythograph.models.llm_client import runtime_status


def main() -> None:
    status = runtime_status()
    print(status["mode"])
    if status["mode"] == "llamacpp":
        print(status["llamacpp_repo_id"])
        print(status["llamacpp_filename"])


if __name__ == "__main__":
    main()
