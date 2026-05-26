# YOLO dataset format

Ultralytics YOLO expects this layout:

```
data/yolo/
├── dataset.yaml              # tells YOLO where everything is
├── images/
│   ├── train/                # *.jpg / *.png
│   └── val/
└── labels/
    ├── train/                # one *.txt per image, same basename
    └── val/
```

Each label file holds one detection per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

All four box numbers are normalized to `[0, 1]` (divide pixel coords by
image width/height). `class_id` is an integer index into the `names` list
in `dataset.yaml`.

**Example label file** (`labels/train/img001.txt`) for an image with a
person and a dog:

```
0 0.512 0.430 0.220 0.640
1 0.780 0.612 0.180 0.290
```

## Quickstart datasets

- **COCO128** — tiny pre-made dataset, great for sanity checking the
  pipeline. Ultralytics auto-downloads it if `data:` is set to `coco128.yaml`.
- **Roboflow Universe** — free YOLO-format datasets across thousands of
  domains (https://universe.roboflow.com).
- **Label your own** — try [Label Studio](https://labelstud.io) or
  [CVAT](https://www.cvat.ai); both export YOLO format directly.

See `dataset.yaml` in this folder for the schema.
