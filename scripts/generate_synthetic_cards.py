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
SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisym.ttf"),
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
    suit_symbol = SUIT_SYMBOLS[suit]
    rank_font = find_font(random.randint(38, 46), bold=True)
    suit_font = find_font(random.randint(30, 38), bold=True)
    small_font = find_font(22, bold=True)

    x = random.randint(15, 23)
    y = random.randint(12, 18)
    draw.text((x, y), rank, font=rank_font, fill=color)
    draw.text((x + 2, y + 48), suit_symbol, font=suit_font, fill=color)

    rotated = Image.new("RGBA", size, (0, 0, 0, 0))
    rdraw = ImageDraw.Draw(rotated)
    rdraw.text((x, y), rank, font=rank_font, fill=color)
    rdraw.text((x + 2, y + 48), suit_symbol, font=suit_font, fill=color)
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
        draw.text((w // 2 - 18, h // 2 - 14), suit_symbol, font=find_font(52, bold=True), fill=color)
    else:
        for yy in range(90, h - 80, 40):
            draw.text((w // 2 - 14, yy), suit_symbol, font=small_font, fill=color)

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


def draw_table_occluder(draw: ImageDraw.ImageDraw, canvas_size, center, card_w: int, card_h: int) -> None:
    """Draw a realistic table object over part of the card after the card is pasted."""
    cx, cy = center
    kind = random.choice(["chip", "card_back", "hand_shadow", "rail"])
    ox = int(cx + random.uniform(-0.35, 0.35) * card_w)
    oy = int(cy + random.uniform(-0.35, 0.35) * card_h)

    if kind == "chip":
        radius = random.randint(max(18, card_w // 12), max(28, card_w // 6))
        fill = random.choice([(230, 230, 225), (35, 35, 38), (180, 30, 40), (40, 70, 170)])
        outline = random.choice([(245, 245, 245), (20, 20, 20)])
        draw.ellipse((ox - radius, oy - radius, ox + radius, oy + radius), fill=fill, outline=outline, width=4)
        inner = int(radius * 0.55)
        draw.ellipse((ox - inner, oy - inner, ox + inner, oy + inner), outline=outline, width=2)
    elif kind == "card_back":
        bw = random.randint(max(55, card_w // 4), max(85, card_w // 2))
        bh = random.randint(max(75, card_h // 4), max(120, card_h // 2))
        color = random.choice([(32, 65, 145), (130, 25, 35), (35, 95, 70)])
        draw.rounded_rectangle((ox - bw // 2, oy - bh // 2, ox + bw // 2, oy + bh // 2), radius=10, fill=color, outline=(235, 235, 235), width=3)
        draw.rectangle((ox - bw // 3, oy - bh // 3, ox + bw // 3, oy + bh // 3), outline=(235, 235, 235), width=2)
    elif kind == "hand_shadow":
        ew = random.randint(max(60, card_w // 4), max(110, card_w // 2))
        eh = random.randint(max(35, card_h // 8), max(70, card_h // 4))
        skin = random.choice([(180, 125, 90), (205, 155, 120), (135, 90, 65)])
        draw.ellipse((ox - ew, oy - eh, ox + ew, oy + eh), fill=skin)
        for i in range(random.randint(2, 4)):
            fx = ox + random.randint(-ew, ew // 2)
            fy = oy + random.randint(-eh, eh)
            draw.rounded_rectangle((fx, fy - 10, fx + random.randint(50, 95), fy + 12), radius=8, fill=skin)
    else:
        rw = random.randint(max(90, card_w // 2), max(180, card_w))
        rh = random.randint(18, 45)
        fill = random.choice([(20, 20, 24), (65, 50, 45), (25, 80, 45)])
        draw.rounded_rectangle((ox - rw // 2, oy - rh // 2, ox + rw // 2, oy + rh // 2), radius=8, fill=fill)


def add_occlusions(bg: Image.Image, canvas_size, center, card_w: int, card_h: int, probability: float) -> Image.Image:
    if random.random() >= probability:
        return bg
    draw = ImageDraw.Draw(bg)
    for _ in range(random.randint(1, 3)):
        draw_table_occluder(draw, canvas_size, center, card_w, card_h)
    return bg


def rotated_corners(cx, cy, w, h, angle_deg):
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    out = []
    for x, y in pts:
        out.append((cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a))
    return out


def transform_points(cx, cy, w, h, angle_deg, points):
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    out = []
    for x, y in points:
        out.append((cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a))
    return out


def local_rect_polygon(card_info, left, top, right, bottom):
    w, h = card_info["w"], card_info["h"]
    points = [
        (left - w / 2, top - h / 2),
        (right - w / 2, top - h / 2),
        (right - w / 2, bottom - h / 2),
        (left - w / 2, bottom - h / 2),
    ]
    return transform_points(card_info["cx"], card_info["cy"], w, h, -card_info["angle"], points)


def mask_polygon(canvas_size, polygon) -> Image.Image:
    mask = Image.new("L", canvas_size, 0)
    ImageDraw.Draw(mask).polygon([(int(x), int(y)) for x, y in polygon], fill=255)
    return mask


def card_polygon(card_info):
    return rotated_corners(card_info["cx"], card_info["cy"], card_info["w"], card_info["h"], -card_info["angle"])


def corner_polygons(card_info):
    w, h = card_info["w"], card_info["h"]
    corner_w = int(w * 0.42)
    corner_h = int(h * 0.36)
    return [
        local_rect_polygon(card_info, 0, 0, corner_w, corner_h),
        local_rect_polygon(card_info, w - corner_w, h - corner_h, w, h),
    ]


def visible_fraction(region_mask: Image.Image, cover_mask: Image.Image) -> float:
    region = np.array(region_mask, dtype=np.uint8) > 0
    area = int(region.sum())
    if area == 0:
        return 0.0
    covered = np.array(cover_mask, dtype=np.uint8) > 0
    visible = region & ~covered
    return float(visible.sum() / area)


def stack_has_readable_corners(cards, canvas_size, min_visible_fraction: float = 0.35) -> bool:
    cover_masks = [mask_polygon(canvas_size, card_polygon(card)) for card in cards]
    later_cover = Image.new("L", canvas_size, 0)
    for idx in range(len(cards) - 1, -1, -1):
        corner_ok = any(
            visible_fraction(mask_polygon(canvas_size, polygon), later_cover) >= min_visible_fraction
            for polygon in corner_polygons(cards[idx])
        )
        if not corner_ok:
            return False
        later_cover = Image.fromarray(np.maximum(np.array(later_cover), np.array(cover_masks[idx])).astype(np.uint8), "L")
    return True


def clamp_center(cx: float, cy: float, card_h: int, canvas_size):
    margin = int(card_h * 0.72)
    return (
        int(max(margin, min(canvas_size[0] - margin, cx))),
        int(max(margin, min(canvas_size[1] - margin, cy))),
    )


def make_card_info(cx: int, cy: int, scale: float, angle: float):
    return {
        "cx": cx,
        "cy": cy,
        "w": int(260 * scale),
        "h": int(360 * scale),
        "angle": angle,
    }


def paste_card(bg: Image.Image, card_info):
    rank = random.choice(RANKS)
    suit = random.choice(SUITS)
    card = draw_card(rank, suit, (card_info["w"], card_info["h"]), random.randint(0, 1000))
    rotated = card.rotate(card_info["angle"], expand=True, resample=Image.Resampling.BICUBIC)
    cx, cy = card_info["cx"], card_info["cy"]
    bg.paste(rotated, (int(cx - rotated.width / 2), int(cy - rotated.height / 2)), rotated)
    return card_info


def paste_random_card(bg: Image.Image, canvas_size, cx: int, cy: int, scale: float, angle: float):
    return paste_card(bg, make_card_info(cx, cy, scale, angle))


def generate_stack_cards(canvas_size, max_stack_cards: int, require_readable_corners: bool = True):
    for _ in range(80):
        count = random.randint(2, max(2, max_stack_cards))
        base_scale = random.uniform(0.78, 1.08)
        base_h = int(360 * base_scale)
        base_cx = random.randint(int(base_h * 0.82), canvas_size[0] - int(base_h * 0.82))
        base_cy = random.randint(int(base_h * 0.82), canvas_size[1] - int(base_h * 0.82))
        base_angle = random.uniform(-72, 72)
        fan_step = random.uniform(8, 22) * random.choice([-1, 1])
        offset_x = random.uniform(48, 96) * random.choice([-1, 1])
        offset_y = random.uniform(22, 62) * random.choice([-1, 1])
        mid = (count - 1) / 2.0
        cards = []
        for idx in range(count):
            delta = idx - mid
            scale = base_scale * random.uniform(0.94, 1.04)
            card_h = int(360 * scale)
            cx, cy = clamp_center(
                base_cx + delta * offset_x + random.uniform(-10, 10),
                base_cy + delta * offset_y + random.uniform(-8, 8),
                card_h,
                canvas_size,
            )
            angle = base_angle + delta * fan_step + random.uniform(-4, 4)
            cards.append(make_card_info(cx, cy, scale, angle))
        if not require_readable_corners or stack_has_readable_corners(cards, canvas_size):
            return cards
    return cards


def make_label_line(card_info, canvas_size) -> str:
    pts = rotated_corners(card_info["cx"], card_info["cy"], card_info["w"], card_info["h"], -card_info["angle"])
    norm = []
    for x, y in pts:
        norm.extend([x / canvas_size[0], y / canvas_size[1]])
    return "0 " + " ".join(f"{v:.6f}" for v in norm)


def save_obb_sample(
    img_path: Path,
    label_path: Path,
    canvas_size=(960, 720),
    occlusion_prob: float = 0.45,
    stack_prob: float = 0.35,
    max_stack_cards: int = 4,
) -> None:
    bg = make_background(canvas_size)
    cards = []
    is_stack = False

    if random.random() < stack_prob:
        is_stack = True
        cards = generate_stack_cards(canvas_size, max_stack_cards, require_readable_corners=True)
        for card in cards:
            paste_card(bg, card)
    else:
        scale = random.uniform(0.75, 1.2)
        card_h = int(360 * scale)
        cx = random.randint(int(card_h * 0.7), canvas_size[0] - int(card_h * 0.7))
        cy = random.randint(int(card_h * 0.7), canvas_size[1] - int(card_h * 0.7))
        angle = random.uniform(-88, 88)
        cards.append(paste_random_card(bg, canvas_size, cx, cy, scale, angle))

    if cards and not is_stack:
        main_card = random.choice(cards)
        bg = add_occlusions(bg, canvas_size, (main_card["cx"], main_card["cy"]), main_card["w"], main_card["h"], occlusion_prob)

    if random.random() < 0.5:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.7)))
    bg.save(img_path, quality=92)

    label_path.write_text("\n".join(make_label_line(card, canvas_size) for card in cards) + "\n", encoding="utf-8")


def save_corner_sample(path: Path, class_name: str, occlusion_prob: float = 0.12) -> None:
    rank, suit = class_name[:-1], class_name[-1]
    card = draw_card(rank, suit, variant=random.randint(0, 1000))
    crop = card.crop((0, 0, 95, 125)).convert("RGB")
    angle = random.uniform(-6, 6)
    crop = crop.rotate(angle, expand=False, fillcolor=(250, 250, 248), resample=Image.Resampling.BICUBIC)
    if random.random() < occlusion_prob:
        draw = ImageDraw.Draw(crop)
        # Keep this mild; fully covered corner labels should be treated as unreadable at runtime.
        x1 = random.randint(45, 82)
        y1 = random.randint(5, 95)
        x2 = min(95, x1 + random.randint(18, 45))
        y2 = min(125, y1 + random.randint(14, 36))
        draw.rounded_rectangle((x1, y1, x2, y2), radius=5, fill=random.choice([(35, 35, 35), (190, 190, 180), (70, 90, 65)]))
    if random.random() < 0.4:
        crop = crop.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.4)))
    crop.save(path, quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--obb-count", type=int, default=3000)
    parser.add_argument("--corner-count", type=int, default=300, help="images per class")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--obb-occlusion-prob", type=float, default=0.45)
    parser.add_argument("--corner-occlusion-prob", type=float, default=0.12)
    parser.add_argument("--stack-prob", type=float, default=0.35)
    parser.add_argument("--max-stack-cards", type=int, default=4)
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
            save_obb_sample(
                obb / "images" / split / f"{stem}.jpg",
                obb / "labels" / split / f"{stem}.txt",
                occlusion_prob=args.obb_occlusion_prob,
                stack_prob=args.stack_prob,
                max_stack_cards=args.max_stack_cards,
            )

    for split in ["train", "val"]:
        per_class = args.corner_count if split == "train" else max(20, args.corner_count // 5)
        for class_name in CLASSES:
            class_dir = cls / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                save_corner_sample(
                    class_dir / f"{class_name}_{i:05d}.jpg",
                    class_name,
                    occlusion_prob=args.corner_occlusion_prob,
                )

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
