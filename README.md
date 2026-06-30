# PokerSight

Two-stage visual pipeline for rotated playing cards:

1. **Card frame detection** with YOLO OBB. It finds the rotated card rectangle anywhere in the camera frame.
2. **Card value recognition** on the perspective-corrected card corner. This keeps the middle artwork/decorations out of the value decision.

The runtime already supports arbitrary card rotation. The included synthetic generator is meant to bootstrap a working model quickly; for production, fine-tune with real camera data from your target table/camera/deck styles.

## Quick start

```bat
cd C:\zhengyiming\强化学习的学习\realtime_card_detector
python scripts\generate_synthetic_cards.py --out data\synthetic --obb-count 3000 --corner-count 300
python scripts\train_detector.py --data data\synthetic\cards_obb.yaml --epochs 50 --imgsz 640 --weights yolo11n-obb.pt --workers 0 --no-amp
python scripts\train_classifier.py --data data\synthetic\corners_cls --epochs 30 --imgsz 96 --weights yolo11n-cls.pt --workers 0 --no-amp
python scripts\realtime_detect.py --detector runs\obb\card_obb\weights\best.pt --classifier runs\classify\corner_cls\weights\best.pt --source 0
```

If the YOLO pretrained weights are not cached locally, Ultralytics will try to download them. Use `--weights path\to\local.pt` in offline environments.

## Synthetic occlusion

The OBB generator includes partial occlusion augmentation by default. It draws table-like objects such as chips, card backs, shadows, and rails over the card while keeping the full card OBB label, so the first stage learns to localize partly covered cards.

The corner classifier uses only mild occlusion augmentation. If all readable corners are blocked, runtime recognition should return low confidence instead of guessing.

```bat
python scripts\generate_synthetic_cards.py --out data\synthetic --obb-occlusion-prob 0.45 --corner-occlusion-prob 0.12
```

## Open-source data strategy

Public playing-card datasets are usually regular bounding boxes or full-card class labels, not OBB plus corner labels. The recommended path is:

- use public card images to add visual diversity;
- convert any full-card bbox to OBB only when the cards are axis-aligned, or label OBB with a tool for rotated cards;
- generate/crop corner labels from rectified cards for the second stage;
- keep a small real validation set from your actual camera setup.

The runtime and training scripts are dataset-agnostic as long as labels are in Ultralytics formats:

- OBB detector: `class x1 y1 x2 y2 x3 y3 x4 y4`, normalized.
- Classifier: folder-per-class image classification dataset, e.g. `train/AS/*.jpg`, `val/AS/*.jpg`.

## Files

- `scripts/generate_synthetic_cards.py`: creates OBB detector data and corner classification data.
- `scripts/train_detector.py`: trains YOLO OBB card-frame detector.
- `scripts/train_classifier.py`: trains corner value classifier.
- `scripts/realtime_detect.py`: webcam/video/image realtime inference.
- `scripts/infer_image.py`: one-image end-to-end inference and annotated output.
- `configs/card_obb.yaml`: template dataset config.

## Current trained baseline in this workspace

The synthetic baseline trained in this workspace is:

```bat
python scripts\realtime_detect.py --detector runs\obb\card_obb_synth\weights\best.pt --classifier runs\classify\corner_cls_synth\weights\best.pt --source 0
```

One-image test:

```bat
python scripts\infer_image.py --detector runs\obb\card_obb_synth\weights\best.pt --classifier runs\classify\corner_cls_synth\weights\best.pt --image data\synthetic\cards_obb\images\val\val_000000.jpg --out runs\demo\val_000000_annotated.jpg
```

## Benchmark

```powershell
python scripts\benchmark_pipeline.py --detector runs\obb\card_obb_synth\weights\best.pt --images data\synthetic\cards_obb\images\val --mode detector --device 0
python scripts\benchmark_pipeline.py --detector runs\obb\card_obb_synth\weights\best.pt --images data\synthetic\cards_obb\images\val --mode template --device 0
python scripts\benchmark_pipeline.py --detector runs\obb\card_obb_synth\weights\best.pt --classifier runs\classify\corner_cls_synth\weights\best.pt --images data\synthetic\cards_obb\images\val --mode classifier --device 0
```

## Notes

- The synthetic generator outputs rank/suit corner labels as text-like cards. This is enough to prove the pipeline and train a baseline, but not enough for robust casino/cardroom deployment.
- Class names use ASCII suit IDs (`S/H/D/C`) for spades/hearts/diamonds/clubs, while generated card images render the actual symbols (`♠/♥/♦/♣`).
- For arbitrary decorative decks, add real corner crops from those decks to `corners_cls`.
- For speed, use `yolo11n-obb` and a small classifier first; export to ONNX/TensorRT later if needed.
