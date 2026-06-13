import base64
import html
import json
import mimetypes
from pathlib import Path

from mythograph.config import ROOT_DIR


GALLERY_PATH = ROOT_DIR / "data" / "waiting_gallery.json"


def render_waiting_gallery(message: str = "") -> str:
    slides = _story_slides()
    slide_html = "\n".join(_render_slide(slide) for slide in slides)
    dots = "".join("<span></span>" for _ in slides)
    return f"""
    <section class="ma-story-carousel">
      <div class="ma-story-header">
        <span>{html.escape(message)}</span>
      </div>
      <div class="ma-story-window">
        <div class="ma-story-track" style="--ma-slide-count: {len(slides)};">
          {slide_html}
        </div>
      </div>
      <div class="ma-story-dots">
        {dots}
      </div>
    </section>
    """


def _story_slides() -> list[dict[str, str]]:
    items = _gallery_items()
    fallback = [
        ("Patience Becoming Power", "Quiet shapes gathering force around a warm center."),
        ("A Spring After Grief", "Soft color returning slowly, like light entering a closed room."),
        ("A Quiet Rebellion", "Small forms refusing the direction of a larger pattern."),
        ("Memory Under Glass", "Blurred fragments preserved inside transparent layers."),
    ]
    slides = []
    for index, item in enumerate(items[:4]):
        default_title, default_caption = fallback[index % len(fallback)]
        slides.append(
            {
                "title": str(item.get("title") or default_title),
                "caption": str(item.get("caption") or item.get("phrase") or default_caption),
                "image": str(item.get("image") or ""),
            }
        )
    return slides or [
        {"title": title, "caption": caption, "image": ""}
        for title, caption in fallback
    ]


def _gallery_items() -> list[dict]:
    try:
        raw = json.loads(GALLERY_PATH.read_text(encoding="utf-8"))
    except Exception:
        raw = []
    return raw if isinstance(raw, list) and raw else []


def _render_slide(slide: dict[str, str]) -> str:
    title = html.escape(slide["title"])
    caption = html.escape(slide["caption"])
    image_html = _image_html(slide["image"], title)
    return f"""
        <article class="ma-story-slide">
          {image_html}
          <div class="ma-story-copy">
            <p class="ma-story-kicker">From the atelier archive</p>
            <h3>{title}</h3>
            <p>{caption}</p>
          </div>
        </article>
        """


def _image_html(path_text: str, title: str) -> str:
    path = (ROOT_DIR / path_text).resolve() if path_text else None
    if path and path.exists() and path.is_file() and _is_inside_workspace(path):
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:{mime};base64,{data}" alt="{title}">'
    return '<div class="ma-story-placeholder"></div>'


def _is_inside_workspace(path: Path) -> bool:
    try:
        path.relative_to(ROOT_DIR)
        return True
    except ValueError:
        return False
