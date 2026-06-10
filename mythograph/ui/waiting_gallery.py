import base64
import html
import json
import mimetypes
from pathlib import Path

from mythograph.config import ROOT_DIR


GALLERY_PATH = ROOT_DIR / "data" / "waiting_gallery.json"


def render_waiting_gallery(message: str) -> str:
    cards = "\n".join(_render_card(item, index) for index, item in enumerate(_gallery_items()))
    return f"""
    <section class="ma-waiting-gallery">
      <div class="ma-waiting-head">
        <div class="ma-kicker">Atelier gallery</div>
        <h3>{html.escape(message)}</h3>
      </div>
      <div class="ma-waiting-track">
        {cards}
      </div>
    </section>
    """


def _gallery_items() -> list[dict]:
    try:
        raw = json.loads(GALLERY_PATH.read_text(encoding="utf-8"))
    except Exception:
        raw = []
    return raw if isinstance(raw, list) and raw else _fallback_items()


def _render_card(item: dict, index: int) -> str:
    title = html.escape(str(item.get("title", f"Study {index + 1}")))
    phrase = html.escape(str(item.get("phrase", "")))
    image_html = _image_html(str(item.get("image", "")), title)
    return f"""
      <article class="ma-waiting-card">
        {image_html}
        <div class="ma-waiting-copy">
          <strong>{title}</strong>
          <p>{phrase}</p>
        </div>
      </article>
    """


def _image_html(path_text: str, title: str) -> str:
    path = (ROOT_DIR / path_text).resolve() if path_text else None
    if path and path.exists() and path.is_file() and _is_inside_workspace(path):
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:{mime};base64,{data}" alt="{title}">'
    return '<div class="ma-waiting-placeholder"></div>'


def _is_inside_workspace(path: Path) -> bool:
    try:
        path.relative_to(ROOT_DIR)
        return True
    except ValueError:
        return False


def _fallback_items() -> list[dict]:
    return [
        {
            "title": "A Small Relief",
            "phrase": "The painting begins where ordinary words become enough.",
            "image": "",
        },
        {
            "title": "A Useful Pause",
            "phrase": "While the model thinks, the gallery keeps the room warm.",
            "image": "",
        },
    ]
