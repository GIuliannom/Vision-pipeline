# Vision Pipeline v3: YOLO + RT-DETR ensemble + PyTorchVideo + MCP + Llama

What v3 adds over v2:

- **RT-DETR** alongside YOLO — a transformer-based detector trained on the
  same dataset (so they share class indices).
- **Score-level ensemble fusion** — when both detectors look at the same
  frame, their per-class probabilities get combined per matched box. The
  final label is the argmax of the combined distribution.
- **Five selectable modes:** `yolo`, `rtdetr`, `ensemble`, `video`, `all`.

## The ensemble in 30 seconds

```
Frame
  │
  ├─► YOLO     → [car 0.50, truck 0.40, bus 0.10]   @ box (100,100,300,300)
  └─► RT-DETR  → [car 0.10, truck 0.60, bus 0.30]   @ box (105,105,295,295)
                   ↓
              IoU matches the boxes (IoU > 0.5)
                   ↓
              Combine per-class (average by default):
              car   = (0.50 + 0.10) / 2 = 0.30
              truck = (0.40 + 0.60) / 2 = 0.50  ← winner
              bus   = (0.10 + 0.30) / 2 = 0.20
                   ↓
              Final: truck (0.50), source: both
```

When a detection is seen by only one of the two detectors, it's kept but
its confidence is multiplied by `ensemble_unmatched_penalty` (default 0.7),
since one-detector evidence is weaker than two-detector agreement.

## What's new (files)

```
models/rtdetr/train.py            # RT-DETR trainer
inference/detector_backend.py     # unified backend for YOLO and RT-DETR
fusion/detector_fusion.py         # IoU matching + score combining
configs/rtdetr.yaml               # RT-DETR training config
scripts/train_rtdetr.py           # entry script
```

`scripts/export_onnx.py` and `inference/webcam.py` were updated to support
RT-DETR. Everything else (video model, MCP, Ollama, alerts) is unchanged
from v2.

## Setup

```bash
unzip vision-pipeline-v3.zip && cd vision-pipeline-v3
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:3b
```

## The critical rule: both detectors MUST share a class list

The ensemble code combines per-class scores by index. If YOLO's class 0 is
"person" and RT-DETR's class 0 is "car," the fusion is meaningless.

To guarantee this, `configs/rtdetr.yaml` points at the same `data:` file
as YOLO (`data/yolo/dataset.yaml`). Don't change that unless you're
prepared to handle the mapping yourself.

The webcam loop checks at startup and warns if the class lists diverge.

## Train

```bash
# 1. YOLO on your data
python scripts/train_yolo.py   --config configs/yolo.yaml

# 2. RT-DETR on the SAME data (same dataset.yaml)
python scripts/train_rtdetr.py --config configs/rtdetr.yaml

# 3. Optional: action recognizer
python scripts/train_video.py  --config configs/video.yaml
```

## Export to ONNX

```bash
python scripts/export_onnx.py --target all
```

Or specifically:
```bash
python scripts/export_onnx.py --target rtdetr
```

## Run

```bash
# Just the ensemble (YOLO + RT-DETR fused)
python scripts/run_all.py --backend ensemble --no-llm --no-alerts

# Everything (ensemble + video + LLM + alerts)
python scripts/run_all.py --backend all

# Just YOLO
python scripts/run_all.py --backend yolo
```

Live key controls in the window:
- `1` → YOLO only
- `2` → RT-DETR only
- `3` → ensemble (YOLO + RT-DETR)
- `4` → video only
- `5` → all (ensemble + video)
- `q` → quit

When the ensemble is active, each box is tagged in its label:
- `✓` — both detectors agreed on this object
- `[A]` — only YOLO saw it (lower confidence)
- `[B]` — only RT-DETR saw it (lower confidence)

## Tuning the ensemble

In `configs/inference.yaml`:

```yaml
ensemble_iou_threshold: 0.5    # how much boxes must overlap to be "the same object"
ensemble_method: average       # average | weighted | multiply | max
ensemble_weight_yolo: 1.0      # only for method=weighted
ensemble_weight_rtdetr: 1.0
ensemble_unmatched_penalty: 0.7  # confidence multiplier for one-detector detections
```

Method picks:
- `average` — both equal say, most common choice
- `weighted` — use when one detector is known to be more accurate
- `multiply` — strictest: only objects both detectors are confident about
  survive (multiplicative confidence)
- `max` — loosest: keeps the highest score per class across detectors

## Honest limits

- **Speed**: running both detectors roughly halves your FPS vs single-detector
  mode. On a decent GPU you'll still see 30+ FPS with `yolov8n` + `rtdetr-l`;
  on CPU the ensemble will be slower. Switch to a smaller RT-DETR variant
  if it's a bottleneck (note RT-DETR doesn't ship a nano version though).
- **Per-class probabilities are approximated.** Ultralytics' default
  output gives a top-1 score per box, not a full pre-NMS distribution. The
  backend constructs a one-hot-with-leftover-spread vector that produces
  the right behavior for your "models pick different top classes"
  scenario, but it's not a true softmax distribution. See the docstring
  in `inference/detector_backend.py` for details.
- **RT-DETR is larger than YOLOv8n.** If you trained YOLO on COCO128 in
  3 minutes, training RT-DETR will take noticeably longer (it's a
  transformer; uses more memory and runs slower per step).
