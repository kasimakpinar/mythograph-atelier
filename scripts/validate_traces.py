import json
from pathlib import Path


def main() -> None:
    path = Path("data/example_traces.jsonl")
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        json.loads(line)
        print(f"line {index}: ok")


if __name__ == "__main__":
    main()
