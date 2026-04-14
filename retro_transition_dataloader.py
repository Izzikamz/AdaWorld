from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass
class TransitionLoaderConfig:
    # DataLoader/runtime knobs.
    batch_size: int = 16
    num_workers: int = 4
    shuffle_train: bool = True
    # Optional resize target for loaded frames (square output).
    image_size: int | None = None
    # Image range convention used by downstream model code.
    normalize: str = "zero_one"  # choices: zero_one, minus_one_one
    pin_memory: bool = True
    drop_last_train: bool = False


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    # Eagerly load JSONL into memory for simple random indexing.
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _build_label_vocab(paths: list[Path]) -> dict[str, int]:
    """Scan all provided JSONL files to build a mapping from action labels to integer IDs."""
    labels = set()
    for path in paths:
        if not path.exists():
            continue
        for row in _load_jsonl(path):
            labels.add(str(row.get("action_label", "UNKNOWN")))
    return {label: index for index, label in enumerate(sorted(labels))}


class RetroTransitionDataset(Dataset):
    """Dataset for retro transition samples stored in transitions JSONL files.

    Expected record fields:
    - frame_t: path to first frame image
    - frame_tp1: path to second frame image
    - action_label: string action label
    - action_vector: list[int] action vector
    - game/system/episode/step: optional metadata
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        label_to_id: dict[str, int],
        image_size: int | None = None,
        normalize: str = "zero_one",
    ) -> None:
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Missing JSONL file: {self.jsonl_path}")

        self.records = _load_jsonl(self.jsonl_path)
        self.root = self.jsonl_path.parent.parent
        self.image_size = image_size
        if normalize not in {"zero_one", "minus_one_one"}:
            raise ValueError("normalize must be one of: zero_one, minus_one_one")
        self.normalize = normalize
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_frame_path(self, frame_path: str) -> Path:
        # Handle absolute paths and repo-relative paths from saved metadata.
        candidate = Path(frame_path)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if candidate.exists():
            return candidate
        joined = self.root / candidate
        if joined.exists():
            return joined
        raise FileNotFoundError(f"Frame path not found: {frame_path}")

    def _load_image(self, image_path: str) -> Tensor:
        image = iio.imread(self._resolve_frame_path(image_path))
        if image.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape={image.shape} for {image_path}")

        # Convert HWC uint8 -> CHW float in [0, 1].
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).contiguous()  # C, H, W

        if self.image_size is not None and (tensor.shape[1] != self.image_size or tensor.shape[2] != self.image_size):
            # Resize with bilinear interpolation (keeps channel-first layout).
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.normalize == "minus_one_one":
            # Optional remap to [-1, 1] for diffusion/world-model style inputs.
            tensor = tensor * 2.0 - 1.0

        return tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        action_label = str(row.get("action_label", "UNKNOWN"))

        frame_t = self._load_image(str(row["frame_t"]))
        frame_tp1 = self._load_image(str(row["frame_tp1"]))

        action_vector = torch.tensor(row.get("action_vector", []), dtype=torch.float32)
        # Unknown labels are kept as -1 so callers can mask or filter.
        action_id = self.label_to_id.get(action_label, -1)

        return {
            "frame_t": frame_t,
            "frame_tp1": frame_tp1,
            "frame_pair": torch.stack([frame_t, frame_tp1], dim=0),
            "action_label": action_label,
            "action_id": torch.tensor(action_id, dtype=torch.long),
            "action_vector": action_vector,
            "game": str(row.get("game", "")),
            "system": str(row.get("system", "")),
            "episode": torch.tensor(int(row.get("episode", 0)), dtype=torch.long),
            "step": torch.tensor(int(row.get("step", 0)), dtype=torch.long),
        }


def create_transition_dataloaders(
    dataset_dir: str | Path,
    config: TransitionLoaderConfig | None = None,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    """Create train/val/test dataloaders from a stage dataset directory.

    Required files under dataset_dir:
    - transitions_train.jsonl
    - transitions_val.jsonl
    - transitions_test.jsonl

    Returns:
    - dataloaders: dict with keys "train", "val", "test" and DataLoader values
    - label_to_id: dict mapping action labels to integer IDs
    """

    cfg = config or TransitionLoaderConfig()
    root = Path(dataset_dir)
    train_jsonl = root / "transitions_train.jsonl"
    val_jsonl = root / "transitions_val.jsonl"
    test_jsonl = root / "transitions_test.jsonl"

    for required in [train_jsonl, val_jsonl, test_jsonl]:
        if not required.exists():
            raise FileNotFoundError(f"Missing split file: {required}")

    label_to_id = _build_label_vocab([train_jsonl, val_jsonl, test_jsonl])

    train_dataset = RetroTransitionDataset(
        jsonl_path=train_jsonl,
        label_to_id=label_to_id,
        image_size=cfg.image_size,
        normalize=cfg.normalize,
    )
    val_dataset = RetroTransitionDataset(
        jsonl_path=val_jsonl,
        label_to_id=label_to_id,
        image_size=cfg.image_size,
        normalize=cfg.normalize,
    )
    test_dataset = RetroTransitionDataset(
        jsonl_path=test_jsonl,
        label_to_id=label_to_id,
        image_size=cfg.image_size,
        normalize=cfg.normalize,
    )

    dataloaders = {
        # Train shuffles by default; val/test are deterministic.
        "train": DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_train,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last_train,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
        ),
    }

    return dataloaders, label_to_id


if __name__ == "__main__":
    # Minimal smoke test: load one batch and print tensor shapes.
    root = Path("data/retro_platformer_stage1_v1")
    dataloaders, label_map = create_transition_dataloaders(root)
    print(f"Loaded dataloaders for: {root}")
    print(f"Num action labels: {len(label_map)}")
    first_batch = next(iter(dataloaders["train"]))
    print("train.batch.frame_t", tuple(first_batch["frame_t"].shape))
    print("train.batch.frame_tp1", tuple(first_batch["frame_tp1"].shape))
    print("train.batch.action_id", tuple(first_batch["action_id"].shape))