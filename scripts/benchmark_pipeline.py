import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))

import cv2
from ultralytics import YOLO

from realtime_detect import ClassifierRecognizer, TemplateCornerRecognizer, warp_card


def collect_images(path: str, limit: int):
    root = Path(path)
    if root.is_file():
        return [root]
    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(sorted(root.glob(pattern)))
    return images[:limit]


def run_pass(detector, recognizer, images, imgsz: int, conf: float, device: str | None, detector_only: bool):
    frames = 0
    cards = 0
    started = time.perf_counter()
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        result = detector.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        frames += 1

        if detector_only or result.obb is None or result.obb.xyxyxyxy is None:
            continue

        for pts in result.obb.xyxyxyxy.cpu().numpy():
            card = warp_card(frame, pts)
            recognizer.predict(card)
            cards += 1

    elapsed = time.perf_counter() - started
    return frames, cards, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True)
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--images", required=True)
    parser.add_argument("--mode", choices=["detector", "template", "classifier"], default="template")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="optional Ultralytics device, e.g. cpu, 0, cuda:0")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    images = collect_images(args.images, args.limit)
    if not images:
        raise RuntimeError(f"No images found: {args.images}")

    detector = YOLO(args.detector)
    detector_only = args.mode == "detector"
    recognizer = None
    if args.mode == "template":
        recognizer = TemplateCornerRecognizer()
    elif args.mode == "classifier":
        if not args.classifier:
            raise RuntimeError("--classifier is required for classifier mode")
        recognizer = ClassifierRecognizer(args.classifier, device=args.device)

    if args.warmup:
        run_pass(detector, recognizer, images[: args.warmup], args.imgsz, args.conf, args.device, detector_only)

    frames, cards, elapsed = run_pass(detector, recognizer, images, args.imgsz, args.conf, args.device, detector_only)
    fps = frames / elapsed if elapsed else 0.0
    cards_per_second = cards / elapsed if elapsed else 0.0
    avg_cards = cards / frames if frames else 0.0
    print(f"mode={args.mode}")
    print(f"frames={frames}")
    print(f"cards={cards}")
    print(f"avg_cards_per_frame={avg_cards:.2f}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"fps={fps:.2f}")
    print(f"cards_per_second={cards_per_second:.2f}")


if __name__ == "__main__":
    main()
