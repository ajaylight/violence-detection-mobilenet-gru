from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    import winsound
except ImportError:  # pragma: no cover - only used on non-Windows platforms.
    winsound = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "violence_mobilenet_gru.keras"
DEFAULT_LOG_PATH = ROOT / "logs" / "incidents.csv"
ALERT_THRESHOLD = 0.80
LOG_COOLDOWN_SECONDS = 30.0
SAMPLE_INTERVAL_SECONDS = 0.30
PREDICT_INTERVAL_SECONDS = 0.90

_alarm_active = False
_alarm_stop_event = threading.Event()
_alarm_thread: threading.Thread | None = None
_last_incident_log_monotonic = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live violence detection with real-time alerts.")
    parser.add_argument("--source", default="0", help="Camera index, video file, or stream URL.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Trained .keras model path.")
    parser.add_argument("--threshold", type=float, default=ALERT_THRESHOLD, help="Alert confidence threshold.")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH, help="CSV incident log path.")
    parser.add_argument("--log-cooldown", type=float, default=LOG_COOLDOWN_SECONDS, help="Seconds between logs.")
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=SAMPLE_INTERVAL_SECONDS,
        help="Seconds between frames added to the model sequence.",
    )
    parser.add_argument(
        "--predict-interval",
        type=float,
        default=PREDICT_INTERVAL_SECONDS,
        help="Seconds between model predictions.",
    )
    return parser.parse_args()


def normalize_source(source: str) -> int | str:
    """Use integer webcam IDs for numeric sources and strings for URLs/files."""
    return int(source) if source.isdigit() else source


def ensure_incident_log(log_path: Path) -> None:
    """Create logs directory and CSV header when the incident file is missing."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Confidence"])


def _fallback_alarm_loop() -> None:
    """Fallback alarm for platforms without winsound: repeated terminal bell."""
    while not _alarm_stop_event.wait(0.8):
        sys.stdout.write("\a")
        sys.stdout.flush()


def play_alarm() -> None:
    """Start a continuous alarm while violence is being detected."""
    global _alarm_active, _alarm_thread

    if _alarm_active:
        return

    _alarm_active = True
    _alarm_stop_event.clear()

    if winsound is not None:
        winsound.PlaySound(
            "SystemExclamation",
            winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_LOOP,
        )
        return

    _alarm_thread = threading.Thread(target=_fallback_alarm_loop, daemon=True)
    _alarm_thread.start()


def stop_alarm() -> None:
    """Stop the active alarm when violence is no longer detected."""
    global _alarm_active, _alarm_thread

    if not _alarm_active:
        return

    _alarm_active = False
    _alarm_stop_event.set()

    if winsound is not None:
        winsound.PlaySound(None, 0)

    if _alarm_thread is not None:
        _alarm_thread.join(timeout=1.0)
        _alarm_thread = None


def log_incident(confidence: float, log_path: Path = DEFAULT_LOG_PATH, cooldown_seconds: float = LOG_COOLDOWN_SECONDS) -> bool:
    """Append one incident row, throttled to avoid duplicate continuous detections."""
    global _last_incident_log_monotonic

    now_monotonic = time.monotonic()
    if _last_incident_log_monotonic and now_monotonic - _last_incident_log_monotonic < cooldown_seconds:
        return False

    ensure_incident_log(log_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, f"{confidence:.4f}"])

    _last_incident_log_monotonic = now_monotonic
    return True


def prepare_frame_for_model(frame: np.ndarray, image_size: int) -> np.ndarray:
    """Convert OpenCV BGR frame to the model's RGB input size."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)


def predict_sequence(model: tf.keras.Model, frame_sequence: deque[np.ndarray]) -> float:
    """Run the existing MobileNet + GRU model on the current live sequence."""
    batch = np.asarray(list(frame_sequence), dtype=np.float32)
    clip = preprocess_input(batch)[None, ...]
    return float(model.predict(clip, verbose=0).reshape(-1)[0])


def draw_alert_overlay(frame: np.ndarray, confidence: float) -> None:
    """Draw a red border and prominent alert message directly on the OpenCV frame."""
    height, width = frame.shape[:2]
    border_width = max(6, width // 140)

    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), border_width)

    banner_height = max(105, height // 6)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Requested alert: "🚨 VIOLENCE DETECTED 🚨".
    # OpenCV's built-in Hershey fonts cannot reliably render emoji, so the on-frame
    # warning keeps the same message in ASCII for dependable display.
    alert_text = "!!! VIOLENCE DETECTED !!!"
    confidence_text = f"Confidence: {confidence * 100:.0f}%"

    cv2.putText(frame, alert_text, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, confidence_text, (24, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)


def draw_normal_overlay(frame: np.ndarray, confidence: float, threshold: float) -> None:
    """Show lightweight status when there is no active alert."""
    status_text = f"Confidence: {confidence * 100:.0f}% | Alert threshold: {threshold * 100:.0f}%"
    cv2.putText(frame, status_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 220, 40), 2, cv2.LINE_AA)


def run_live_alert(args: argparse.Namespace) -> None:
    ensure_incident_log(args.log_path)

    model = tf.keras.models.load_model(str(args.model))
    _, sequence_length, image_size, _, _ = model.input_shape

    capture = cv2.VideoCapture(normalize_source(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open live source: {args.source}")

    frame_sequence: deque[np.ndarray] = deque(maxlen=sequence_length)
    last_sample_time = 0.0
    last_predict_time = 0.0
    confidence = 0.0
    violence_detected = False

    print("Live alert detection started. Press q to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Live source ended or frame could not be read.")
                break

            now = time.monotonic()

            if now - last_sample_time >= args.sample_interval:
                frame_sequence.append(prepare_frame_for_model(frame, image_size))
                last_sample_time = now

            if len(frame_sequence) == sequence_length and now - last_predict_time >= args.predict_interval:
                confidence = predict_sequence(model, frame_sequence)
                violence_detected = confidence >= args.threshold
                last_predict_time = now

                if violence_detected:
                    play_alarm()
                    log_incident(confidence, args.log_path, args.log_cooldown)
                else:
                    stop_alarm()

            if violence_detected:
                draw_alert_overlay(frame, confidence)
            else:
                draw_normal_overlay(frame, confidence, args.threshold)

            cv2.imshow("Live Violence Detection Alerts", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_alarm()
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    run_live_alert(parse_args())


if __name__ == "__main__":
    main()
