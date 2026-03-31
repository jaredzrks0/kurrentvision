from pathlib import Path

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

from src.modeling.datasets.base import (
    collect_raw_samples,
    collect_synthetic_samples,
    filter_raw_samples,
    train_test_split,
)


class RawDataset(Dataset):
    """Dataset for XML-annotated data (bounding-box crops) returning raw PIL images for TrOCR."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"])
        image.load()
        image = image.convert("RGB")
        x0, y0, x1, y1 = sample["bbox"]
        crop = image.crop((x0, y0, x1, y1)).copy()
        return {"image": crop, "text": sample["text"], "dataset": sample["dataset"]}


class SyntheticDataset(Dataset):
    """Dataset for synthetic image/text pairs returning raw PIL images for TrOCR."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"])
        image.load()
        image = image.convert("RGB")
        return {"image": image, "text": sample["text"], "dataset": "synthetic"}


# ── Collation ────────────────────────────────────────────────────────────────

def _collate(batch: list[dict]) -> dict:
    return {
        "image": [item["image"] for item in batch],
        "text": [item["text"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
    }


# ── Dataloader Builders ──────────────────────────────────────────────────────

def build_dataloaders(
    root_dir: str | Path,
    exclude: list[str] | None = None,
    synthetic_dir: str | Path | None = None,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders returning raw PIL images for TrOCR.
    Optionally include synthetic data by passing synthetic_dir.
    Returns (train_loader, val_loader, test_loader).
    """
    raw_samples = collect_raw_samples(root_dir, exclude=exclude)
    filtered = filter_raw_samples(raw_samples)

    synth_samples = collect_synthetic_samples(synthetic_dir) if synthetic_dir else []

    raw_train, raw_val, raw_test = train_test_split(filtered, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)
    synth_train, synth_val, synth_test = train_test_split(synth_samples, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed) if synth_samples else ([], [], [])

    train_ds = _combine_datasets(raw_train, synth_train)
    val_ds = _combine_datasets(raw_val, synth_val)
    test_ds = _combine_datasets(raw_test, synth_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
    return train_loader, val_loader, test_loader


def build_synthetic_dataloaders(
    synthetic_dir: str | Path,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from synthetic data only for TrOCR.
    Returns (train_loader, val_loader, test_loader).
    """
    all_samples = collect_synthetic_samples(synthetic_dir)
    train_samples, val_samples, test_samples = train_test_split(all_samples, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)

    train_ds = SyntheticDataset(train_samples)
    val_ds = SyntheticDataset(val_samples)
    test_ds = SyntheticDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
    return train_loader, val_loader, test_loader


def _combine_datasets(raw: list[dict], synth: list[dict]):
    """Create a ConcatDataset from raw + synthetic sample lists."""
    parts = []
    if raw:
        parts.append(RawDataset(raw))
    if synth:
        parts.append(SyntheticDataset(synth))
    if len(parts) == 1:
        return parts[0]
    return ConcatDataset(parts)
