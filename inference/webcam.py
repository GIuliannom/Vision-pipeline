"""
v3 webcam loop.

Backends:
    yolo      - YOLO detector only
    rtdetr    - RT-DETR detector only
    ensemble  - YOLO + RT-DETR fused (combined per-class scores)
    video     - PyTorchVideo action classifier only
    all       - YOLO + RT-DETR ensemble + video
    both      - alias for 'all' (v1/v2 compatibility)

Live keys in the window:
    1 = YOLO
    2 = RT-DETR
    3 = ensemble (YOLO + RT-DETR)
    4 = video only
    5 = all
    q = quit
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from inference.detector_backend import DetectorBackend
from inference.video_backend import VideoClassifier
from fusion.fusion import fuse
from fusion.detector_fusion import fuse_detectors
from utils.draw import draw_detections, draw_actions, draw_hud
from shared_features.scene_state import SCENE, SceneSnapshot


VALID_BACKENDS = {"yolo", "rtdetr", "ensemble", "video", "all", "both"}


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_labels(path: str) -> list[str]:
    out: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def _pick_weights(pt: str, onnx: str, prefer_onnx: bool) -> Path | None:
    pt_p, onnx_p = Path(pt), Path(onnx)
    if prefer_onnx and onnx_p.exists():
        return onnx_p
    if pt_p.exists():
        return pt_p
    if onnx_p.exists():
        return onnx_p
    return None


def build_yolo(cfg: dict) -> DetectorBackend | None:
    w = _pick_weights(cfg["yolo_weights"], cfg["yolo_onnx"], cfg.get("prefer_onnx", True))
    if w is None:
        print(f"[yolo] no weights — disabling.")
        return None
    print(f"[yolo] loading {w}")
    return DetectorBackend(w, kind="yolo",
                           conf=cfg["yolo_conf"], iou=cfg["yolo_iou"], imgsz=cfg["yolo_imgsz"])


def build_rtdetr(cfg: dict) -> DetectorBackend | None:
    w = _pick_weights(cfg["rtdetr_weights"], cfg["rtdetr_onnx"], cfg.get("prefer_onnx", True))
    if w is None:
        print(f"[rtdetr] no weights — disabling.")
        return None
    print(f"[rtdetr] loading {w}")
    return DetectorBackend(w, kind="rtdetr",
                           conf=cfg["rtdetr_conf"], iou=cfg["rtdetr_iou"], imgsz=cfg["rtdetr_imgsz"])


def build_video(cfg: dict) -> VideoClassifier | None:
    w = _pick_weights(cfg["video_weights"], cfg["video_onnx"], cfg.get("prefer_onnx", True))
    if w is None:
        print(f"[video] no weights — disabling.")
        return None
    labels = load_labels(cfg["video_labels_file"])
    print(f"[video] loading {w} with {len(labels)} labels")
    return VideoClassifier(
        weights_path=w, labels=labels,
        clip_len=cfg["video_clip_len"],
        sampling_rate=cfg["video_sampling_rate"],
        side_size=cfg["video_side_size"],
        crop_size=cfg["video_crop_size"],
        topk=cfg["video_topk"],
        prefer_onnx=cfg.get("prefer_onnx", True),
    )


def _check_class_lists_match(yolo: DetectorBackend, rtdetr: DetectorBackend) -> None:
    if yolo.names != rtdetr.names:
        print("[ensemble] ⚠ YOLO and RT-DETR class lists differ!")
        print(f"          YOLO:   {dict(list(yolo.names.items())[:5])}...")
        print(f"          RT-DETR:{dict(list(rtdetr.names.items())[:5])}...")
        print("          Ensemble results may be incorrect. Retrain RT-DETR on the same dataset.")


def run(
    config_path: str,
    backend_override: str | None = None,
    camera_override: int | None = None,
    headless: bool = False,
) -> None:
    cfg = load_config(config_path)
    if camera_override is not None:
        cfg["camera_index"] = camera_override

    backend = backend_override or cfg.get("default_backend", "all")
    if backend == "both":
        backend = "all"
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Backend {backend!r} not in {VALID_BACKENDS}")

    yolo = build_yolo(cfg)
    rtdetr = build_rtdetr(cfg)
    video = build_video(cfg)

    if yolo and rtdetr:
        _check_class_lists_match(yolo, rtdetr)

    # Validate the requested backend is available
    needs = {
        "yolo": [yolo],
        "rtdetr": [rtdetr],
        "ensemble": [yolo, rtdetr],
        "video": [video],
        "all": [yolo or rtdetr, video],   # need at least one detector + video
    }[backend]
    if not all(needs):
        raise RuntimeError(f"Backend {backend!r} needs models that aren't available. "
                           f"yolo={bool(yolo)}, rtdetr={bool(rtdetr)}, video={bool(video)}")

    cap = cv2.VideoCapture(cfg["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_height"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cfg['camera_index']}")

    if not headless:
        cv2.namedWindow("vision-pipeline", cv2.WINDOW_NORMAL)

    last_actions: list[dict] = []
    action_every_n_frames = max(1, cfg.get("video_clip_len", 8) // 2)
    frame_idx = 0
    t_prev = time.time()
    fps = 0.0

    # Ensemble settings
    ens_iou = cfg.get("ensemble_iou_threshold", 0.5)
    ens_method = cfg.get("ensemble_method", "average")
    ens_w_yolo = cfg.get("ensemble_weight_yolo", 1.0)
    ens_w_rtdetr = cfg.get("ensemble_weight_rtdetr", 1.0)
    ens_penalty = cfg.get("ensemble_unmatched_penalty", 0.7)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            dets: list[dict] = []
            actions: list[dict] = []

            run_yolo = backend in ("yolo", "ensemble", "all") and yolo is not None
            run_rtdetr = backend in ("rtdetr", "ensemble", "all") and rtdetr is not None
            run_video = backend in ("video", "all") and video is not None

            dets_yolo = yolo.predict(frame) if run_yolo else []
            dets_rtdetr = rtdetr.predict(frame) if run_rtdetr else []

            if run_yolo and run_rtdetr:
                # Ensemble: combine per-class scores
                names = yolo.names    # already validated same as rtdetr.names
                dets = fuse_detectors(
                    dets_yolo, dets_rtdetr, names,
                    iou_threshold=ens_iou,
                    method=ens_method,
                    weight_a=ens_w_yolo,
                    weight_b=ens_w_rtdetr,
                    unmatched_penalty=ens_penalty,
                )
            elif run_yolo:
                dets = dets_yolo
            elif run_rtdetr:
                dets = dets_rtdetr

            if run_video:
                video.update(frame)
                if frame_idx % action_every_n_frames == 0 and video.ready():
                    last_actions = video.predict()
                actions = last_actions

            fused = fuse(
                dets if (run_yolo or run_rtdetr) else None,
                actions if run_video else None,
                mode=cfg.get("fusion_mode", "concat"),
            )

            now = time.time()
            dt = now - t_prev
            t_prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

            # Detections are published in our normalized form; strip the
            # heavy class_probs np.ndarrays before sharing (not JSON-safe)
            shareable_objects = [
                {k: v for k, v in d.items() if k != "class_probs"}
                for d in fused["objects"]
            ]

            SCENE.update(SceneSnapshot(
                timestamp=now,
                objects=shareable_objects,
                action=fused["action"],
                action_topk=fused["action_topk"],
                summary=fused["scene_summary"],
                backend=backend,
                fps=fps,
            ))

            if not headless:
                if dets:
                    draw_detections(frame, dets)
                if actions:
                    draw_actions(frame, actions)
                draw_hud(
                    frame, backend=backend, fps=fps,
                    summary=fused["scene_summary"] if cfg.get("show_fps", True) else "",
                    show_help=cfg.get("show_help_overlay", True),
                )
                cv2.imshow("vision-pipeline", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("1") and yolo:
                    backend = "yolo"; print(f"[switch] → yolo")
                elif key == ord("2") and rtdetr:
                    backend = "rtdetr"; print(f"[switch] → rtdetr")
                elif key == ord("3") and yolo and rtdetr:
                    backend = "ensemble"; print(f"[switch] → ensemble")
                elif key == ord("4") and video:
                    backend = "video"; print(f"[switch] → video")
                elif key == ord("5"):
                    backend = "all"; print(f"[switch] → all")

            frame_idx += 1
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/inference.yaml")
    p.add_argument("--backend",
                   choices=sorted(VALID_BACKENDS),
                   default=None)
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()
    run(args.config, args.backend, args.camera, args.headless)
