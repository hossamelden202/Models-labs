import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def base_image(seed, size=(64, 64), shift=0, mode="RGB"):
    rng = np.random.default_rng(seed)
    grid = rng.integers(40, 180, (12, 12, 3), dtype=np.uint8)
    img = Image.fromarray(grid).resize(size, Image.Resampling.BICUBIC)
    if shift:
        arr = np.clip(np.asarray(img, dtype=np.int16) + shift, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img.convert(mode) if mode != "RGB" else img


def build_synthetic_dataset(root):
    root = Path(root)

    def put(rel, seed, **kwargs):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        base_image(seed, **kwargs).save(path)

    def copy(src, dst):
        (root / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / src, root / dst)

    for i in range(8):
        put(f"train/GORE/g{i:02d}.png", 100 + i)
        put(f"train/BLOOD/b{i:02d}.png", 200 + i)
    for i in range(40):
        put(f"train/NEUTRAL/n{i:02d}.png", 300 + i)
    put("train/BLOOD/wide.png", 250, size=(400, 40))
    put("train/NEUTRAL/gray.png", 340, mode="L")
    put("train/NEUTRAL/scene_alpha_001.png", 341)

    for i in range(3):
        put(f"val/GORE/v_g{i}.png", 400 + i)
        put(f"val/BLOOD/v_b{i}.png", 410 + i)
        put(f"val/NEUTRAL/v_n{i}.png", 420 + i)
        put(f"test/GORE/t_g{i}.png", 500 + i)
        put(f"test/BLOOD/t_b{i}.png", 510 + i)
        put(f"test/NEUTRAL/t_n{i}.png", 520 + i)
    put("val/GORE/rgba.png", 403, mode="RGBA")
    put("val/NEUTRAL/near_n05.png", 305, shift=10)
    put("test/NEUTRAL/scene_alpha_001.png", 342)

    copy("train/GORE/g00.png", "train/GORE/g00_copy.png")
    copy("train/GORE/g02.png", "train/NEUTRAL/conflict_g2.png")
    copy("train/BLOOD/b01.png", "test/BLOOD/leak_exact_b01.png")

    put("val/BLOOD/truncated.jpg", 413)
    data = (root / "val/BLOOD/truncated.jpg").read_bytes()
    (root / "val/BLOOD/truncated.jpg").write_bytes(data[: len(data) // 2])
    (root / "train/GORE/corrupt.png").write_bytes(b"not an image at all")

    (root / "train/GORE/notes.txt").write_text("not an image")
    (root / "train/.hidden").write_text("hidden")
    put("train/GORE/.ipynb_checkpoints/x.png", 999)
    (root / "train/EMPTY").mkdir()
    return root
