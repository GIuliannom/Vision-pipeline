"""
Real-time webcam inference.

Streams from a webcam, runs the selected backend(s), and overlays results.
Press 1/2/3 to switch backends live; press q to quit.

Backend choices:
    yolo  - object detection per frame
    video - action classification on a rolling clip
    both  - run both and fuse the outputs
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from inference.yolo_backend import YoloDetector
from inference.video_backend import VideoClassifier
from fusion.fusion import fuse
from utils.draw import draw_detections, draw_actions, draw_hud


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_labels(path: str) -> list[str]:
    out: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def build_yolo(cfg: dict) -> YoloDetector | None:
    # Prefer ONNX path if present and prefer_onnx is True
    onnx_path = Path(cfg["yolo_onnx"])
    pt_path = Path(cfg["yolo_weights"])
    if cfg.get("prefer_onnx", True) and onnx_path.exists():
        weights = onnx_path
    elif pt_path.exists():
        weights = pt_path
    elif onnx_path.exists():
        weights = onnx_path
    else:
        print(f"[yolo] no weights found at {pt_path} or {onnx_path} — disabling.")
        return None
    print(f"[yolo] loading {weights}")
    return YoloDetector(
        weights_path=weights,
        conf=cfg["yolo_conf"],
        iou=cfg["yolo_iou"],
        imgsz=cfg["yolo_imgsz"],
        prefer_onnx=cfg.get("prefer_onnx", True),
    )


def build_video(cfg: dict) -> VideoClassifier | None:
    onnx_path = Path(cfg["video_onnx"])
    pt_path = Path(cfg["video_weights"])
    if cfg.get("prefer_onnx", True) and onnx_path.exists():
        weights = onnx_path
    elif pt_path.exists():
        weights = pt_path
    elif onnx_path.exists():
        weights = onnx_path
    else:
        print(f"[video] no weights found at {pt_path} or {onnx_path} — disabling.")
        return None
    labels = load_labels(cfg["video_labels_file"])
    print(f"[video] loading {weights} with {len(labels)} labels")
    return VideoClassifier(
        weights_path=weights,
        labels=labels,
        clip_len=cfg["video_clip_len"],
        sampling_rate=cfg["video_sampling_rate"],
        side_size=cfg["video_side_size"],
        crop_size=cfg["video_crop_size"],
        topk=cfg["video_topk"],
        prefer_onnx=cfg.get("prefer_onnx", True),
    )


def run(config_path: str, backend_override: str | None) -> None:
    cfg = load_config(config_path)
    backend = backend_override or cfg.get("default_backend", "both")

    yolo = build_yolo(cfg)
    video = build_video(cfg)

    if backend == "yolo" and yolo is None:
        raise RuntimeError("YOLO requested but no weights available.")
    if backend == "video" and video is None:
        raise RuntimeError("Video requested but no weights available.")
    if backend == "both" and yolo is None and video is None:
        raise RuntimeError("Neither backend has weights. Train at least one first.")

    cap = cv2.VideoCapture(cfg["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_height"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cfg['camera_index']}")

    cv2.namedWindow("vision-pipeline", cv2.WINDOW_NORMAL)

    last_actions: list[dict] = []
    action_every_n_frames = max(1, cfg.get("video_clip_len", 8) // 2)
    frame_idx = 0
    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            dets: list[dict] = []
            actions: list[dict] = []

            run_yolo = backend in ("yolo", "both") and yolo is not None
            run_video = backend in ("video", "both") and video is not None

            if run_yolo:
                dets = yolo.predict(frame)

            if run_video:
                video.update(frame)
                # Don't run the heavy 3D model every frame
                if frame_idx % action_every_n_frames == 0 and video.ready():
                    last_actions = video.predict()
                actions = last_actions

            fused = fuse(
                dets if run_yolo else None,
                actions if run_video else None,
                mode=cfg.get("fusion_mode", "concat"),
            )

            if run_yolo:
                draw_detections(frame, dets)
            if run_video:
                draw_actions(frame, actions)

            # FPS
            now = time.time()
            dt = now - t_prev
            t_prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

            draw_hud(
                frame,
                backend=backend,
                fps=fps,
                summary=fused["scene_summary"] if cfg.get("show_fps", True) else "",
                show_help=cfg.get("show_help_overlay", True),
            )

            cv2.imshow("vision-pipeline", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1") and yolo is not None:
                backend = "yolo"
                print("[switch] backend -> yolo")
            elif key == ord("2") and video is not None:
                backend = "video"
                print("[switch] backend -> video")
            elif key == ord("3") and (yolo is not None or video is not None):
                backend = "both"
                print("[switch] backend -> both")

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/inference.yaml")
    p.add_argument(
        "--backend",
        choices=["yolo", "video", "both"],
        default=None,
        help="Override default backend from config.",
    )
    args = p.parse_args()
    run(args.config, args.backend)
