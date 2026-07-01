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
    parser.add_argument("--iou", type=float, default=0.7, help="OBB NMS IoU threshold")
    parser.add_argument("--value-conf", type=float, default=0.75)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    detector = YOLO(args.detector)
    recognizer = ClassifierRecognizer(args.classifier, device=args.device) if args.classifier else TemplateCornerRecognizer()
    result = detector.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
    detections = 0
    records = 0
    if result.obb is not None and result.obb.xyxyxyxy is not None:
        for idx, pts in enumerate(result.obb.xyxyxyxy.cpu().numpy()):
            card = warp_card(frame, pts)
            label, value_conf = recognizer.predict(card)
            readable = bool(label) and value_conf >= args.value_conf
            draw_detection(frame, pts, label, value_conf, readable=readable, index=idx)
            detections += 1
            records += int(readable)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"detections={detections} records={records} out={out}")


if __name__ == "__main__":
    main()
