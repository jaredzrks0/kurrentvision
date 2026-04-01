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
    """Dataset for Scraped Kurrent Texts. Note this differs from base model as the ViT is pretrained to just take the raw image"""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"])
        image.load()
        image = image.convert("RGB")

        # Crop to xml given bounding box
        x0, y0, x1, y1 = sample["bbox"]
        crop = image.crop((x0, y0, x1, y1)).copy()
        return {"image": crop, "text": sample["text"], "dataset": sample["dataset"]}


class SyntheticDataset(Dataset):
    """Dataset for synthetic images. Note this differes bc no cropping is needed"""

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


### UTILS FUNCTIONS ###

def _collate(batch: list[dict]) -> dict:
    return {
        "image": [item["image"] for item in batch],
        "text": [item["text"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
    }


def build_dataloaders(
    root_dir: str | Path | None = None,
    exclude: list[str] | None = None,
    synthetic_dir: str | Path | None = None,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    if root_dir is None and synthetic_dir is None:
        raise ValueError("At least one of root_dir or synthetic_dir must be provided.")

    filtered = []
    if root_dir is not None:
        raw_samples = collect_raw_samples(root_dir, exclude=exclude)
        filtered = filter_raw_samples(raw_samples)

    synth_samples = collect_synthetic_samples(synthetic_dir) if synthetic_dir else []

    raw_train, raw_val, raw_test = train_test_split(filtered, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed) if filtered else ([], [], [])
    synth_train, synth_val, synth_test = train_test_split(synth_samples, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed) if synth_samples else ([], [], [])

    train_ds = _combine_datasets(raw_train, synth_train)
    val_ds = _combine_datasets(raw_val, synth_val)
    test_ds = _combine_datasets(raw_test, synth_test)

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
