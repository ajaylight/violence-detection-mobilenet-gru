from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict violence probability for video files.")
    parser.add_argument("--model", type=Path, required=True, help="Trained .keras model path.")
    parser.add_argument("--video", type=Path, required=True, help="Video file or folder.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    parser.add_argument("--windows", type=int, default=8, help="Number of temporal windows to scan per video.")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=3.0,
        help="Approximate seconds covered by each prediction window.",
    )
    return parser.parse_args()


def collect_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in VIDEO_EXTENSIONS)


def read_video_window(
    capture: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    sequence_length: int,
    image_size: int,
) -> np.ndarray:
    positions = np.linspace(start_frame, max(start_frame, end_frame), sequence_length).astype(int)
    frames: list[np.ndarray] = []
    fallback = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if not ok or frame is None:
            frame_rgb = fallback.copy()
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
            fallback = frame_rgb
        frames.append(frame_rgb)

    batch = np.asarray(frames, dtype=np.float32)
    return preprocess_input(batch)


def read_video_windows(
    path: Path,
    sequence_length: int,
    image_size: int,
    windows: int,
    window_seconds: float,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    safe_total = max(1, total_frames)
    window_frames = int(max(sequence_length, round(fps * window_seconds)))

    if safe_total <= window_frames or windows <= 1:
        starts = [0]
    else:
        max_start = max(0, safe_total - window_frames)
        starts = np.linspace(0, max_start, windows).astype(int).tolist()

    video_windows = [
        read_video_window(
            capture,
            start_frame=start,
            end_frame=min(safe_total - 1, start + window_frames),
            sequence_length=sequence_length,
            image_size=image_size,
        )
        for start in starts
    ]

    capture.release()
    return np.asarray(video_windows, dtype=np.float32)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    model = tf.keras.models.load_model(str(args.model))
    _, sequence_length, image_size, _, _ = model.input_shape

    videos = collect_videos(args.video)
    if not videos:
        raise SystemExit(f"No videos found at {args.video}")

    for video in videos:
        clips = read_video_windows(
            video,
            sequence_length,
            image_size,
            windows=args.windows,
            window_seconds=args.window_seconds,
        )
        probabilities = model.predict(clips, verbose=0).reshape(-1)
        probability = float(np.max(probabilities))
        mean_probability = float(np.mean(probabilities))
        label = "VIOLENCE" if probability >= args.threshold else "NON_VIOLENCE"
        print(
            f"{label}\tmax={probability:.4f}\tmean={mean_probability:.4f}\t"
            f"windows={len(probabilities)}\t{video}"
        )


if __name__ == "__main__":
    main()
