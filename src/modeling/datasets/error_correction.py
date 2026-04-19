import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TextPairDataset(Dataset):
    def __init__(self, ocr_preds: list[str], ground_truths: list[str]):
        self.ocr = ocr_preds
        self.clean = ground_truths

    def __len__(self):
        return len(self.ocr)

    def __getitem__(self, idx):
        return {"noisy": self.ocr[idx], "clean": self.clean[idx]}


def _collate(batch):
    return {"noisy": [x["noisy"] for x in batch], "clean": [x["clean"] for x in batch]}


def build_text_pair_dataloaders(
    precomputed: dict[str, tuple[list[str], list[str]]],
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loaders = []
    for split in ("train", "val", "test"):
        ocr_preds, ground_truths = precomputed[split]
        ds = TextPairDataset(ocr_preds, ground_truths)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers, collate_fn=_collate))
    return tuple(loaders)


@torch.no_grad()
def precompute_ocr_predictions(ocr_model, ocr_processor, image_loaders: dict, device) -> dict[str, tuple[list[str], list[str]]]:
    result = {}
    for split, loader in image_loaders.items():
        ocr_preds, ground_truths = [], []
        for batch in tqdm(loader, desc=f"  {split}"):
            pixel_values = ocr_processor(images=batch["image"], return_tensors="pt").pixel_values.to(device)
            generated_ids = ocr_model.generate(pixel_values)
            ocr_preds.extend(ocr_processor.batch_decode(generated_ids, skip_special_tokens=True))
            ground_truths.extend(batch["text"])
        result[split] = (ocr_preds, ground_truths)
    return result
