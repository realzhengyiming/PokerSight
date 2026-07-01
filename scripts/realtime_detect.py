import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]
CLASSES = [rank + suit for suit in SUITS for rank in RANKS]
SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_card(frame: np.ndarray, pts: np.ndarray, out_size=(260, 360)) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    dst = np.array(
        [[0, 0], [out_size[0] - 1, 0], [out_size[0] - 1, out_size[1] - 1], [0, out_size[1] - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, matrix, out_size)


class TemplateCornerRecognizer:
    def __init__(self, size=(96, 128)):
        self.size = size
        self.templates = []
        for class_name in CLASSES:
            self.templates.append((class_name, self._render_template(class_name)))

    def _font(self, size):
        for path in ["C:/Windows/Fonts/seguisym.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/calibrib.ttf"]:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        return ImageFont.load_default()

    def _render_template(self, class_name: str) -> np.ndarray:
        rank, suit = class_name[:-1], class_name[-1]
        img = Image.new("L", self.size, 255)
        draw = ImageDraw.Draw(img)
        draw.text((11, 7), rank, font=self._font(40), fill=0)
        draw.text((13, 55), SUIT_SYMBOLS[suit], font=self._font(34), fill=0)
        return self._prep(np.array(img))

    def _prep(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, self.size)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img.astype(np.float32) / 255.0

    def predict(self, card_bgr: np.ndarray):
        best_label, best_score = None, 1e9
        for k in range(4):
            view = np.rot90(card_bgr, k).copy()
            h, w = view.shape[:2]
            corner = view[: int(h * 0.35), : int(w * 0.38)]
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
            sample = self._prep(gray)
            for label, tmpl in self.templates:
                score = float(np.mean((sample - tmpl) ** 2))
                if score < best_score:
                    best_score = score
                    best_label = label
        confidence = max(0.0, min(1.0, 1.0 - best_score * 4.0))
        return best_label, confidence


class ClassifierRecognizer:
    def __init__(self, weights: str, imgsz: int = 96, device: str | None = None):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.device = device

    def predict(self, card_bgr: np.ndarray):
        best_label, best_conf = None, -1.0
        corners = []
        for k in range(4):
            view = np.rot90(card_bgr, k).copy()
            h, w = view.shape[:2]
            corners.append(view[: int(h * 0.35), : int(w * 0.38)])
        results = self.model.predict(corners, imgsz=self.imgsz, device=self.device, verbose=False)
        for result in results:
            probs = result.probs
            conf = float(probs.top1conf)
            label = result.names[int(probs.top1)]
            if conf > best_conf:
                best_label, best_conf = label, conf
        return best_label, best_conf


def draw_detection(frame, pts, label, conf, readable=True):
    pts_i = pts.astype(np.int32)
    color = (40, 220, 40) if readable else (150, 150, 150)
    cv2.polylines(frame, [pts_i], True, color, 2, cv2.LINE_AA)
    x, y = pts_i[:, 0].min(), pts_i[:, 1].min()
    if readable and label:
        text = f"{label} {conf:.2f}"
    elif label:
        text = f"unreadable {conf:.2f}"
    else:
        text = "card"
    cv2.putText(frame, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def open_source(source):
    if str(source).isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True, help="YOLO OBB detector weights")
    parser.add_argument("--classifier", default=None, help="optional YOLO classification weights")
    parser.add_argument("--source", default="0", help="camera index, video path, or stream URL")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--value-conf", type=float, default=0.75, help="minimum corner value confidence to record a card")
    parser.add_argument("--device", default=None, help="optional Ultralytics device, e.g. cpu, 0, cuda:0")
    parser.add_argument("--save", default=None, help="optional output video path")
    args = parser.parse_args()

    detector = YOLO(args.detector)
    recognizer = ClassifierRecognizer(args.classifier, device=args.device) if args.classifier else TemplateCornerRecognizer()
    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    writer = None
    last = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = detector.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        if result.obb is not None and result.obb.xyxyxyxy is not None:
            polys = result.obb.xyxyxyxy.cpu().numpy()
            for pts in polys:
                card = warp_card(frame, pts)
                label, value_conf = recognizer.predict(card)
                readable = bool(label) and value_conf >= args.value_conf
                draw_detection(frame, pts, label, value_conf, readable=readable)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        cv2.putText(frame, f"FPS {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if args.save:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, 30, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        cv2.imshow("card detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
