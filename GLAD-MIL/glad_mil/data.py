from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SlideRecord:
    slide_id: str
    split: str
    label: int
    feature_path: Path


def read_manifest(path: str | Path) -> list[SlideRecord]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"slide_id", "split", "label", "feature_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
    records = []
    for row in rows:
        split = row["split"].strip().lower()
        if split == "validation":
            split = "val"
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {row['split']!r} for {row['slide_id']}")
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        records.append(SlideRecord(row["slide_id"], split, int(row["label"]), feature_path))
    return records


def split_records(records: list[SlideRecord]) -> dict[str, list[SlideRecord]]:
    result = {name: [record for record in records if record.split == name] for name in ("train", "val", "test")}
    if not result["train"] or not result["val"]:
        raise ValueError("Manifest requires non-empty train and val splits.")
    return result


def load_features(record: SlideRecord, device: torch.device) -> torch.Tensor:
    if record.feature_path.suffix.lower() == ".npy":
        value = np.load(record.feature_path)
    else:
        value = torch.load(record.feature_path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        for key in ("features", "feature", "tensor"):
            if key in value:
                value = value[key]
                break
        else:
            raise ValueError(f"No feature tensor key found in {record.feature_path}")
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim != 2 or value.shape[0] == 0:
        raise ValueError(f"Features for {record.slide_id} must have shape [instances, dimensions].")
    return value.to(device)
