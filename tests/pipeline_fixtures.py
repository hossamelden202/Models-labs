import io
import zipfile
from pathlib import Path

from PIL import Image

BRIGHT = {"RED": (200, 50, 40), "GREEN": (40, 200, 50), "BLUE": (40, 50, 200)}
NAMES = list(BRIGHT)


def dark(color):
    return tuple(round(v * 0.3) for v in color)


def png(color):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def build_dark_dataset(root, per_group=12):
    root = Path(root)
    for cls, color in BRIGHT.items():
        for tone, value in (("bright", color), ("dark", dark(color))):
            for i in range(per_group):
                path = root / cls / f"{tone}_{i:03d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png(value))
    return root


def dark_zip(per_group=12):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for cls, color in BRIGHT.items():
            for tone, value in (("bright", color), ("dark", dark(color))):
                for i in range(per_group):
                    archive.writestr(f"{cls}/{tone}_{i:03d}.png", png(value))
    return buffer.getvalue()
