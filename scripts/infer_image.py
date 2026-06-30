import argparse
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))

import cv2
from ultralytics import YOLO

from realtime_detect import ClassifierRecognizer, TemplateCornerRecognizer, draw_detection, warp_card


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True)
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    detector = YOLO(args.detector)
    recognizer = ClassifierRecognizer(args.classifier) if args.classifier else TemplateCornerRecognizer()
    result = detector.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    detections = 0
    if result.obb is not None and result.obb.xyxyxyxy is not None:
        for pts in result.obb.xyxyxyxy.cpu().numpy():
            card = warp_card(frame, pts)
            label, value_conf = recognizer.predict(card)
            draw_detection(frame, pts, label, value_conf)
            detections += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"detections={detections} out={out}")


if __name__ == "__main__":
    main()
