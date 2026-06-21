from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LABEL_FOLDERS = {
    "non_violence": 0,
    "nonviolence": 0,
    "normal": 0,
    "violence": 1,
    "violent": 1,
    "fight": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a video violence detector.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root folder.")
    parser.add_argument("--output", type=Path, default=Path("../models/violence_mobilenet_gru.keras"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument(
        "--max-videos-per-class",
        type=int,
        default=0,
        help="Optional cap for quick smoke tests or fast demo training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-imagenet", action="store_true", help="Do not load ImageNet weights.")
    return parser.parse_args()


def collect_videos(dataset_root: Path) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []

    for folder in dataset_root.iterdir():
        if not folder.is_dir():
            continue
        label = LABEL_FOLDERS.get(folder.name.lower())
        if label is None:
            continue

        for path in folder.rglob("*"):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(path)
                labels.append(label)

    if not paths:
        raise SystemExit(
            "No videos found. Expected folders like dataset/violence and dataset/non_violence."
        )

    return paths, labels


def limit_videos_per_class(
    paths: list[Path],
    labels: list[int],
    max_videos_per_class: int,
    seed: int,
) -> tuple[list[Path], list[int]]:
    if max_videos_per_class <= 0:
        return paths, labels

    rng = random.Random(seed)
    by_class: dict[int, list[Path]] = {}
    for path, label in zip(paths, labels):
        by_class.setdefault(label, []).append(path)

    limited_paths: list[Path] = []
    limited_labels: list[int] = []
    for label, class_paths in sorted(by_class.items()):
        rng.shuffle(class_paths)
        for path in class_paths[:max_videos_per_class]:
            limited_paths.append(path)
            limited_labels.append(label)

    combined = list(zip(limited_paths, limited_labels))
    rng.shuffle(combined)
    return [item[0] for item in combined], [item[1] for item in combined]


def read_video_frames(path: Path, sequence_length: int, image_size: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    fallback = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    if total_frames <= 0:
        positions = [0] * sequence_length
    else:
        positions = np.linspace(0, max(0, total_frames - 1), sequence_length).astype(int)

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

    capture.release()
    batch = np.asarray(frames, dtype=np.float32)
    return preprocess_input(batch)


class VideoSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        batch_size: int,
        sequence_length: int,
        image_size: int,
        shuffle: bool,
        augment: bool,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_ids = self.indices[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        videos = []
        targets = []

        for idx in batch_ids:
            video = read_video_frames(self.paths[idx], self.sequence_length, self.image_size)
            if self.augment and random.random() < 0.5:
                video = video[:, :, ::-1, :]
            videos.append(video)
            targets.append(self.labels[idx])

        return np.asarray(videos, dtype=np.float32), np.asarray(targets, dtype=np.float32)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_model(
    sequence_length: int,
    image_size: int,
    learning_rate: float,
    use_imagenet: bool,
) -> tf.keras.Model:
    inputs = layers.Input(shape=(sequence_length, image_size, image_size, 3))
    base = MobileNetV2(
        include_top=False,
        weights="imagenet" if use_imagenet else None,
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False

    x = layers.TimeDistributed(base)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.LayerNormalization()(x)
    x = layers.GRU(128, dropout=0.3)(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    paths, labels = collect_videos(args.dataset)
    paths, labels = limit_videos_per_class(paths, labels, args.max_videos_per_class, args.seed)
    print(f"Using {len(paths)} videos: {sum(labels)} violence, {len(labels) - sum(labels)} non-violence")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=labels,
    )

    train_data = VideoSequence(
        train_paths,
        train_labels,
        args.batch_size,
        args.sequence_length,
        args.image_size,
        shuffle=True,
        augment=True,
    )
    val_data = VideoSequence(
        val_paths,
        val_labels,
        args.batch_size,
        args.sequence_length,
        args.image_size,
        shuffle=False,
        augment=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(
        args.sequence_length,
        args.image_size,
        args.learning_rate,
        use_imagenet=not args.no_imagenet,
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.output),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(str(args.output.with_suffix(".csv"))),
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(str(args.output))
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()

