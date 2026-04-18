from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from modeling.constants import PAD_TOKEN, UNK_TOKEN, END_TOKEN
from modeling.datasets.base import (
    collect_raw_samples,
    collect_synthetic_samples,
    filter_raw_samples,
    build_vocab,
    image_transforms,
    train_test_split,
)


class RawDataset(Dataset):
    """Dataset for XML-annotated data (bounding-box crops) fed to the basic model."""

    def __init__(self, samples: list[dict], vocab: dict[str, int], max_len: int, transform=None):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"])
        image.load()
        image = image.convert("RGB")
        x0, y0, x1, y1 = sample["bbox"]
        crop = image.crop((x0, y0, x1, y1)).copy()
        crop = self.transform(crop)
        target = _encode_text(sample["text"], self.vocab, self.max_len)
        return {"image": crop, "target": target, "text": sample["text"], "dataset": sample["dataset"]}


class SyntheticDataset(Dataset):
    """Dataset for synthetic image/text pairs fed to the basic model."""

    def __init__(self, samples: list[dict], vocab: dict[str, int], max_len: int, transform=None):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"])
        image.load()
        image = image.convert("RGB")
        image = self.transform(image)
        target = _encode_text(sample["text"], self.vocab, self.max_len)
        return {"image": image, "target": target, "text": sample["text"], "dataset": "synthetic"}


# Shared Helpers

def _encode_text(text: str, vocab: dict[str, int], max_len: int) -> torch.Tensor:
    """Encode text to a padded tensor of vocab indices."""
    ids = [vocab.get(char, vocab[UNK_TOKEN]) for char in text]
    ids = ids[:max_len]
    ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    ids += [vocab[END_TOKEN]]
    return torch.tensor(ids, dtype=torch.long)


def _collate(batch: list[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    texts = [item["text"] for item in batch]
    datasets = [item["dataset"] for item in batch]
    return {"image": images, "target": targets, "text": texts, "dataset": datasets}


# Dataloader Builders

def build_dataloaders(
    root_dir: str | Path | None = None,
    exclude: list[str] | None = None,
    synthetic_dir: str | Path | None = None,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int], int]:
    """Build train/val/test DataLoaders for the basic model.
    input root_dir for raw data, synthetic_dir for synthetic data, 
    """
    if root_dir is None and synthetic_dir is None:
        raise ValueError("At least one of root_dir or synthetic_dir must be provided.")

    filtered = []
    if root_dir is not None:
        raw_samples = collect_raw_samples(root_dir, exclude=exclude)
        filtered = filter_raw_samples(raw_samples)

    synth_samples = collect_synthetic_samples(synthetic_dir) if synthetic_dir else []

    all_samples = filtered + synth_samples
    vocab = build_vocab(all_samples)
    max_len = max(len(s["text"]) for s in all_samples)

    raw_train, raw_val, raw_test = train_test_split(filtered, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed) if filtered else ([], [], [])
    synth_train, synth_val, synth_test = train_test_split(synth_samples, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed) if synth_samples else ([], [], [])

    tfm = image_transforms()

    train_ds = _combine_datasets(raw_train, synth_train, vocab, max_len, tfm)
    val_ds = _combine_datasets(raw_val, synth_val, vocab, max_len, tfm)
    test_ds = _combine_datasets(raw_test, synth_test, vocab, max_len, tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
    return train_loader, val_loader, test_loader, vocab, max_len


def _combine_datasets(raw: list[dict], synth: list[dict], vocab, max_len, tfm):
    """Create a ConcatDataset from raw + synthetic sample lists."""
    from torch.utils.data import ConcatDataset
    parts = []
    if raw:
        parts.append(RawDataset(raw, vocab, max_len, transform=tfm))
    if synth:
        parts.append(SyntheticDataset(synth, vocab, max_len, transform=tfm))
    if len(parts) == 1:
        return parts[0]
    return ConcatDataset(parts)
