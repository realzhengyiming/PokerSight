import argparse
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]
RED_SUITS = {"H", "D"}
CLASSES = [rank + suit for suit in SUITS for rank in RANKS]


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def draw_card(rank: str, suit: str, size=(260, 360), variant: int = 0) -> Image.Image:
    w, h = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = 18
    fill = random.choice([(255, 255, 250, 255), (248, 248, 245, 255), (252, 252, 255, 255)])
    outline = random.choice([(20, 20, 20, 255), (80, 80, 80, 255), (150, 150, 150, 255)])
    draw.rounded_rectangle((2, 2, w - 3, h - 3), radius=radius, fill=fill, outline=outline, width=3)

    color = (190, 20, 35, 255) if suit in RED_SUITS else (15, 15, 20, 255)
    rank_font = find_font(random.randint(38, 46), bold=True)
    suit_font = find_font(random.randint(30, 38), bold=True)
    small_font = find_font(22, bold=True)

    x = random.randint(15, 23)
    y = random.randint(12, 18)
    draw.text((x, y), rank, font=rank_font, fill=color)
    draw.text((x + 2, y + 48), suit, font=suit_font, fill=color)

    rotated = Image.new("RGBA", size, (0, 0, 0, 0))
    rdraw = ImageDraw.Draw(rotated)
    rdraw.text((x, y), rank, font=rank_font, fill=color)
    rdraw.text((x + 2, y + 48), suit, font=suit_font, fill=color)
    rotated = rotated.rotate(180)
    img.alpha_composite(rotated)

    # Add harmless center decorations so the classifier learns to ignore them.
    if variant % 3 == 0:
        for _ in range(random.randint(2, 7)):
            cx = random.randint(70, w - 70)
            cy = random.randint(100, h - 100)
            rr = random.randint(8, 22)
            draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=color, width=2)
    elif variant % 3 == 1:
        draw.text((w // 2 - 18, h // 2 - 14), suit, font=find_font(52, bold=True), fill=color)
    else:
        for yy in range(90, h - 80, 40):
            draw.text((w // 2 - 14, yy), suit, font=small_font, fill=color)

    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.5)))
    return img


def make_background(size) -> Image.Image:
    w, h = size
    base = np.zeros((h, w, 3), dtype=np.uint8)
    color = np.array(random.choice([(30, 95, 55), (50, 70, 90), (95, 75, 55), (45, 45, 45)]), dtype=np.uint8)
    noise = np.random.normal(0, 10, (h, w, 3)).astype(np.int16)
    base[:] = color
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(base, "RGB")


def rotated_corners(cx, cy, w, h, angle_deg):
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    out = []
    for x, y in pts:
        out.append((cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a))
    return out


def save_obb_sample(img_path: Path, label_path: Path, canvas_size=(960, 720)) -> None:
    bg = make_background(canvas_size)
    rank = random.choice(RANKS)
    suit = random.choice(SUITS)
    scale = random.uniform(0.75, 1.2)
    card_w, card_h = int(260 * scale), int(360 * scale)
    card = draw_card(rank, suit, (card_w, card_h), random.randint(0, 1000))
    angle = random.uniform(-88, 88)
    cx = random.randint(int(card_h * 0.7), canvas_size[0] - int(card_h * 0.7))
    cy = random.randint(int(card_h * 0.7), canvas_size[1] - int(card_h * 0.7))
    rotated = card.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    bg.paste(rotated, (int(cx - rotated.width / 2), int(cy - rotated.height / 2)), rotated)

    if random.random() < 0.5:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.7)))
    bg.save(img_path, quality=92)

    pts = rotated_corners(cx, cy, card_w, card_h, -angle)
    norm = []
    for x, y in pts:
        norm.extend([x / canvas_size[0], y / canvas_size[1]])
    label_path.write_text("0 " + " ".join(f"{v:.6f}" for v in norm) + "\n", encoding="utf-8")


def save_corner_sample(path: Path, class_name: str) -> None:
    rank, suit = class_name[:-1], class_name[-1]
    card = draw_card(rank, suit, variant=random.randint(0, 1000))
    crop = card.crop((0, 0, 95, 125)).convert("RGB")
    angle = random.uniform(-6, 6)
    crop = crop.rotate(angle, expand=False, fillcolor=(250, 250, 248), resample=Image.Resampling.BICUBIC)
    if random.random() < 0.4:
        crop = crop.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.4)))
    crop.save(path, quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--obb-count", type=int, default=3000)
    parser.add_argument("--corner-count", type=int, default=300, help="images per class")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out).resolve()
    obb = out / "cards_obb"
    cls = out / "corners_cls"
    reset_dir(obb)
    reset_dir(cls)

    for split in ["train", "val"]:
        (obb / "images" / split).mkdir(parents=True, exist_ok=True)
        (obb / "labels" / split).mkdir(parents=True, exist_ok=True)

    val_count = int(args.obb_count * args.val_ratio)
    train_count = args.obb_count - val_count
    for split, count in [("train", train_count), ("val", val_count)]:
        for i in range(count):
            stem = f"{split}_{i:06d}"
            save_obb_sample(obb / "images" / split / f"{stem}.jpg", obb / "labels" / split / f"{stem}.txt")

    for split in ["train", "val"]:
        per_class = args.corner_count if split == "train" else max(20, args.corner_count // 5)
        for class_name in CLASSES:
            class_dir = cls / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                save_corner_sample(class_dir / f"{class_name}_{i:05d}.jpg", class_name)

    yaml_text = (
        f"path: {obb.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: card\n"
    )
    (out / "cards_obb.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Wrote {obb} and {cls}")


if __name__ == "__main__":
    main()
