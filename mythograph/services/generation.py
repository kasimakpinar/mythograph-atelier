import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from mythograph.config import OUTPUT_DIR
from mythograph.schemas.art_recipe import ArtRecipe


def generate_fallback_image(recipe: ArtRecipe, seed: int | None = None, size: int = 1024) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_value = seed if seed is not None else abs(hash(recipe.image_prompt)) % 10_000_000
    rng = random.Random(seed_value)

    img = Image.new("RGB", (size, size), recipe.palette[0])
    draw = ImageDraw.Draw(img, "RGBA")

    _paint_background(draw, recipe.palette, size, rng)
    _paint_symbol_shapes(draw, recipe, size, rng)
    _paint_direction_lines(draw, recipe.palette, size, rng)

    img = img.filter(ImageFilter.SMOOTH_MORE)
    texture = _make_texture(size, rng)
    img = Image.blend(img, texture, 0.12)

    path = OUTPUT_DIR / f"mythograph_{seed_value}.png"
    img.save(path)
    return str(path)


def _paint_background(draw: ImageDraw.ImageDraw, palette: list[str], size: int, rng: random.Random) -> None:
    for _ in range(55):
        x = rng.randint(-80, size)
        y = rng.randint(-80, size)
        w = rng.randint(80, 320)
        h = rng.randint(60, 260)
        color = _hex_to_rgba(rng.choice(palette), rng.randint(28, 82))
        draw.rectangle([x, y, x + w, y + h], fill=color)


def _paint_symbol_shapes(draw: ImageDraw.ImageDraw, recipe: ArtRecipe, size: int, rng: random.Random) -> None:
    center = size // 2
    for index, symbol in enumerate(recipe.symbols):
        color = _hex_to_rgba(recipe.palette[(index + 1) % len(recipe.palette)], 150)
        name = symbol.visual.lower()
        offset = index * 110 - 160
        if "door" in name:
            draw.rounded_rectangle([center + offset, 210, center + offset + 120, 800], radius=8, outline=color, width=18)
        elif "flame" in name:
            points = [(center + offset, 240), (center + offset + 90, 560), (center + offset - 40, 820), (center + offset - 130, 560)]
            draw.polygon(points, fill=color)
        elif "mountain" in name:
            draw.polygon([(90, 820), (center + offset, 220), (930, 820)], outline=color, width=24)
        elif "mirror" in name:
            draw.ellipse([center + offset - 95, 245, center + offset + 95, 775], outline=color, width=20)
        elif "storm" in name:
            for step in range(8):
                y = 230 + step * 70
                draw.arc([120 + offset, y, 900 + offset, y + 180], 200, 345, fill=color, width=12)
        else:
            draw.line([120, center + offset, 900, center - offset], fill=color, width=18)


def _paint_direction_lines(draw: ImageDraw.ImageDraw, palette: list[str], size: int, rng: random.Random) -> None:
    accent = _hex_to_rgba(palette[min(3, len(palette) - 1)], 210)
    for i in range(4):
        angle = rng.uniform(-0.7, 0.7)
        y = int(size * (0.32 + i * 0.12))
        x_shift = int(math.sin(angle) * 180)
        draw.line([90, y, size - 90, y + x_shift], fill=accent, width=rng.randint(5, 12))


def _make_texture(size: int, rng: random.Random) -> Image.Image:
    texture = Image.new("RGB", (size, size), "#f2efe8")
    pixels = texture.load()
    for y in range(size):
        for x in range(size):
            grain = rng.randint(-12, 12)
            base = max(0, min(255, 238 + grain))
            pixels[x, y] = (base, base - 3, base - 8)
    return texture


def _hex_to_rgba(value: str, alpha: int) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), alpha
