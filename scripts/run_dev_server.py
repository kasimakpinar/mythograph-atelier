import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import demo


demo.queue(default_concurrency_limit=2).launch(
    server_name="127.0.0.1",
    server_port=7860,
    prevent_thread_lock=True,
)

while True:
    time.sleep(3600)
