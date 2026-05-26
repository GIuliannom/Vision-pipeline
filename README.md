# Vision Pipeline: YOLO + PyTorchVideo

A dual-model pipeline that trains an object detector (YOLO) and an action
recognizer (PyTorchVideo) independently, exports both to ONNX for a unified
inference runtime, and runs them live against a webcam — with an
ensemble/fusion layer that combines their outputs into one structured answer
about what's in the frame.

## What each model does

| Model | Input | Output | Question it answers |
|---|---|---|---|
| **YOLO** (Ultralytics) | Single frame | Bounding boxes + classes | *What objects are in the frame?* |
| **PyTorchVideo** | Clip (N frames) | Action class + score | *What is happening in the scene?* |

They solve different problems, so "shared learning" here means **ensemble
fusion at inference time** plus **a unified ONNX runtime**, not weight sharing
(which doesn't make sense across these architectures).

## Project layout

```
vision-pipeline/
├── configs/             # YAML hyperparameter configs
├── data/
│   ├── yolo/            # YOLO-format dataset (images + labels)
│   └── video/           # PyTorchVideo dataset (clips by class folder)
├── models/
│   ├── yolo/            # YOLO training script
│   └── video/           # PyTorchVideo training script
├── export/              # ONNX export for both models
├── inference/           # Real-time webcam inference (backend-selectable)
├── fusion/              # Ensemble logic combining outputs
├── scripts/             # Convenience entry points
└── utils/               # Shared helpers (drawing, I/O, etc.)
```

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (either or both)
python scripts/train_yolo.py  --config configs/yolo.yaml
python scripts/train_video.py --config configs/video.yaml

# 3. Export both to ONNX
python scripts/export_onnx.py --target yolo
python scripts/export_onnx.py --target video

# 4. Run live webcam with selectable backend
python scripts/run_webcam.py --backend yolo
python scripts/run_webcam.py --backend video
python scripts/run_webcam.py --backend both     # fusion mode
```

## Backend selection (live)

While the webcam window is running, you can press:

- `1` → YOLO only
- `2` → PyTorchVideo only
- `3` → Both (ensemble)
- `q` → quit

## Data formats

See `data/yolo/README.md` and `data/video/README.md` for the exact layout
each trainer expects, with examples.

## Notes on the "ONNX-style shared learning"

ONNX is a model **format**, not a way to merge models. What this pipeline
does:

1. Trains each model on its own data with its own loss.
2. Exports both to ONNX so they share one runtime (`onnxruntime`).
3. Fuses outputs at inference time: YOLO says *what's there*, video model says
   *what's happening*, the fusion layer produces a single structured result
   like `{"objects": [...], "action": "...", "confidence": ...}`.

For true weight-sharing (e.g. a shared backbone with two heads) you'd need a
custom architecture, which is a much larger project — happy to scope it
separately.
