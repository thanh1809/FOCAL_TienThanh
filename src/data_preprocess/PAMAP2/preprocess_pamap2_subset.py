"""Create and preprocess a subject-disjoint PAMAP2 subset for FOCAL.

The original FOCAL repository contains the generic dataloader/model code but
does not contain a PAMAP2 parser or PAMAP2 YAML configuration.  This script:

1. Copies the selected raw subjects into a separate folder.
2. Uses the wrist IMU only, matching the FOCAL paper's PAMAP2 experiment.
3. Builds 2-second windows at 100 Hz, with 0.4-second intervals and 50%%
   interval overlap.
4. Saves time-domain tensors in the structure expected by the FOCAL loader.
5. Writes subject-disjoint train/validation/test index files.

Default subset:
    subject103 -> train/validation
    subject109 -> test

The two subjects contain about 11.8%% of the raw PAMAP2 rows, which is the
closest practical subject-disjoint subset to 10%% for this dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ACTIVITY_IDS = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

# FOCAL expects contiguous class IDs for the classification head.
LABEL_MAP = {activity_id: idx for idx, activity_id in enumerate(sorted(ACTIVITY_IDS))}

SAMPLE_RATE = 100
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
WINDOW_STEP = WINDOW_SIZE  # no overlap between 2-second samples, as in Table 1
INTERVAL_SECONDS = 0.4
INTERVAL_SIZE = int(SAMPLE_RATE * INTERVAL_SECONDS)
INTERVAL_STEP = INTERVAL_SIZE // 2  # 50% overlap

# PAMAP2 has: timestamp, activity ID, HR, then three 17-column IMU blocks.
# The first block is the dominant-arm wrist IMU.
WRIST_BLOCK_START = 3
WRIST_ACC = slice(WRIST_BLOCK_START + 1, WRIST_BLOCK_START + 4)
WRIST_GYRO = slice(WRIST_BLOCK_START + 7, WRIST_BLOCK_START + 10)
WRIST_MAG = slice(WRIST_BLOCK_START + 10, WRIST_BLOCK_START + 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset"),
        help="PAMAP2_Dataset/PAMAP2_Dataset directory containing Protocol/ and Optional/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/PAMAP2_1of10"),
        help="Processed FOCAL dataset output directory.",
    )
    parser.add_argument(
        "--raw-subset-root",
        type=Path,
        default=Path("pamap2_1of10_raw"),
        help="Separate raw-subset output directory.",
    )
    parser.add_argument("--train-subjects", nargs="+", default=["subject103"])
    parser.add_argument("--test-subjects", nargs="+", default=["subject109"])
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Tail fraction of each training session used for validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the generated subset/output folders before rebuilding them.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def copy_raw_subset(input_root: Path, raw_subset_root: Path, subjects: set[str], overwrite: bool) -> None:
    if raw_subset_root.exists() and overwrite:
        shutil.rmtree(raw_subset_root)
    raw_subset_root.mkdir(parents=True, exist_ok=True)

    for split in ("Protocol", "Optional"):
        source_dir = input_root / split
        if not source_dir.exists():
            continue
        target_dir = raw_subset_root / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for source_file in sorted(source_dir.glob("subject*.dat")):
            if source_file.stem not in subjects:
                continue
            shutil.copy2(source_file, target_dir / source_file.name)


def load_raw_file(path: Path) -> np.ndarray:
    # The official files are whitespace-separated and contain NaN values.
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 54:
        raise ValueError(f"Unexpected PAMAP2 shape in {path}: {data.shape}")
    return data[:, :54]


def fill_nan(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    for column in range(values.shape[1]):
        column_values = values[:, column]
        finite = np.isfinite(column_values)
        if not finite.any():
            column_values[:] = 0.0
            continue
        if not finite.all():
            valid_indices = np.flatnonzero(finite)
            missing_indices = np.flatnonzero(~finite)
            column_values[missing_indices] = np.interp(
                missing_indices, valid_indices, column_values[valid_indices]
            )
    return values


def intervalize(values: np.ndarray) -> np.ndarray:
    """Return [channels, intervals, samples_per_interval]."""
    intervals = []
    for start in range(0, WINDOW_SIZE - INTERVAL_SIZE + 1, INTERVAL_STEP):
        interval = values[start : start + INTERVAL_SIZE]
        if len(interval) == INTERVAL_SIZE:
            intervals.append(interval)
    return np.stack(intervals, axis=0).transpose(2, 0, 1).astype(np.float32)


def majority_label(labels: np.ndarray) -> tuple[int, float, int] | None:
    labels = labels.astype(np.int64)
    labels = labels[labels != 0]
    if labels.size == 0:
        return None
    label, count = Counter(labels.tolist()).most_common(1)[0]
    purity = count / labels.size
    if label not in LABEL_MAP or purity < 0.9:
        return None
    return LABEL_MAP[label], purity, int(label)


def normalize_modalities(modalities: list[np.ndarray]) -> list[np.ndarray]:
    """Normalize each sensor axis per window to keep scales comparable."""
    normalized = []
    for values in modalities:
        mean = np.nanmean(values, axis=(0, 1), keepdims=True)
        std = np.nanstd(values, axis=(0, 1), keepdims=True)
        std = np.maximum(std, 1e-6)
        normalized.append((values - mean) / std)
    return normalized


def process_file(
    path: Path,
    output_dir: Path,
    subject: str,
    split: str,
) -> list[dict]:
    data = fill_nan(load_raw_file(path))
    records: list[dict] = []
    for window_start in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_STEP):
        window = data[window_start : window_start + WINDOW_SIZE]
        label_info = majority_label(window[:, 1])
        if label_info is None:
            continue

        modalities = [
            window[:, WRIST_ACC],
            window[:, WRIST_GYRO],
            window[:, WRIST_MAG],
        ]
        modalities = normalize_modalities(modalities)
        tensor_data = {
            "wrist": {
                "acc": torch.from_numpy(intervalize(modalities[0])),
                "gyro": torch.from_numpy(intervalize(modalities[1])),
                "mag": torch.from_numpy(intervalize(modalities[2])),
            }
        }

        sample_id = len(records)
        output_file = output_dir / "samples" / f"{subject}_{split}_{sample_id:06d}.pt"
        torch.save(
            {
                "label": torch.tensor(label_info[0], dtype=torch.long),
                "activity_id": label_info[2],
                "subject": subject,
                "session": split,
                "window_start": window_start,
                "data": tensor_data,
                "flag": {"wrist": {"acc": True, "gyro": True, "mag": True}},
            },
            output_file,
        )
        records.append(
            {
                "path": output_file,
                "subject": subject,
                "session": split,
                "window_start": window_start,
                "label": label_info[0],
            }
        )
    return records


def write_index(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            # Relative paths keep the generated dataset portable to Colab.
            handle.write(record["relative_path"].replace("\\", "/") + "\n")


def main() -> None:
    args = parse_args()
    input_root = resolve(args.input_root)
    output_root = resolve(args.output_root)
    raw_subset_root = resolve(args.raw_subset_root)
    train_subjects = set(args.train_subjects)
    test_subjects = set(args.test_subjects)
    selected_subjects = train_subjects | test_subjects

    if train_subjects & test_subjects:
        raise ValueError("Train and test subjects must be disjoint.")
    if not input_root.exists():
        raise FileNotFoundError(f"PAMAP2 input root does not exist: {input_root}")
    if not 0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    (output_root / "samples").mkdir(parents=True, exist_ok=True)
    copy_raw_subset(input_root, raw_subset_root, selected_subjects, args.overwrite)

    all_records: list[dict] = []
    source_files = []
    for split in ("Protocol", "Optional"):
        source_files.extend(sorted((input_root / split).glob("subject*.dat")))
    source_files = [f for f in source_files if f.stem in selected_subjects]
    if not source_files:
        raise FileNotFoundError(f"No selected subject files found under {input_root}")

    for source_file in source_files:
        records = process_file(source_file, output_root, source_file.stem, source_file.parent.name)
        all_records.extend(records)
        print(f"Processed {source_file.name} ({source_file.parent.name}): {len(records)} windows")

    # The original README runs train.py from the src directory.  Write paths
    # relative to that directory so the generated index files work both on
    # Windows and after uploading the repository to Colab.
    repo_root = Path.cwd()
    runtime_root = repo_root / "src"
    for record in all_records:
        record["relative_path"] = Path(os.path.relpath(record["path"], runtime_root)).as_posix()

    train_records = [r for r in all_records if r["subject"] in train_subjects]
    test_records = [r for r in all_records if r["subject"] in test_subjects]

    # Validation is the last part of each training session, while test remains
    # completely subject-disjoint.
    train_split: list[dict] = []
    val_split: list[dict] = []
    for subject in sorted(train_subjects):
        for session in sorted({r["session"] for r in train_records if r["subject"] == subject}):
            session_records = sorted(
                [r for r in train_records if r["subject"] == subject and r["session"] == session],
                key=lambda r: r["window_start"],
            )
            val_count = max(1, math.ceil(len(session_records) * args.val_ratio))
            train_split.extend(session_records[:-val_count])
            val_split.extend(session_records[-val_count:])

    index_dir = output_root / "indexes"
    write_index(index_dir / "pretrain_index.txt", train_split)
    write_index(index_dir / "train_index.txt", train_split)
    write_index(index_dir / "val_index.txt", val_split)
    write_index(index_dir / "test_index.txt", test_records)

    manifest = {
        "selected_subjects": sorted(selected_subjects),
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
        "raw_rows_approximation": "subject103 + subject109 = 11.8% of original raw rows",
        "sample_rate_hz": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "interval_seconds": INTERVAL_SECONDS,
        "interval_overlap": 0.5,
        "location": "wrist",
        "modalities": ["acc", "gyro", "mag"],
        "activity_ids": ACTIVITY_IDS,
        "label_map": LABEL_MAP,
        "counts": {
            "all": len(all_records),
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_records),
        },
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest["counts"], indent=2))
    print(f"Raw subset: {raw_subset_root}")
    print(f"Processed dataset: {output_root}")


if __name__ == "__main__":
    main()
