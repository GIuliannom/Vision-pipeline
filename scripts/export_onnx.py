"""Entry point: export trained models to ONNX."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from export.export_onnx import export_yolo, export_rtdetr, export_video  # noqa: E402

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--target",
        choices=["yolo", "rtdetr", "video", "all"],
        default="all",
    )
    p.add_argument("--yolo-config", default="configs/yolo.yaml")
    p.add_argument("--rtdetr-config", default="configs/rtdetr.yaml")
    p.add_argument("--video-config", default="configs/video.yaml")
    args = p.parse_args()

    if args.target in ("yolo", "all"):
        export_yolo(args.yolo_config)
    if args.target in ("rtdetr", "all"):
        export_rtdetr(args.rtdetr_config)
    if args.target in ("video", "all"):
        export_video(args.video_config)
