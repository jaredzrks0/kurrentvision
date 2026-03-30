from pathlib import Path
from xml.etree import ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.modeling.constants import IMG_WIDTH, IMG_HEIGHT, PAD_TOKEN, END_TOKEN, UNK_TOKEN

class KurrentDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        vocab: dict[str, int],
        max_len: int,
        transform=None,
    ):
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
        target = self.encode_text(sample["text"], self.vocab, self.max_len)
        return {"image": crop, "target": target, "text": sample["text"], "dataset": sample["dataset"]}
    
    def encode_text(self, text: str, vocab: dict[str, int], max_len: int) -> torch.Tensor:
        """Encode text to a padded tensor of vocab indices."""
        ids = [vocab.get(char, vocab[UNK_TOKEN]) for char in text]
        
        # If longer than the longest allowed, cut off. Else pad
        ids = ids[:max_len]
        ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))

        # Finally append an end token
        ids += [vocab[END_TOKEN]]

        return torch.tensor(ids, dtype=torch.long)



class TrOCRDataset(Dataset):
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


class DataloaderBuilder():

    def __init__(self):
        self.ALTO_NAMESPACE = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
        self.PAGE_NAMESPACE = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH

    ##### ALTO PARSER FUNCTIONS #####
    def parse_alto(self, xml_path: Path) -> list[tuple[tuple, str, tuple[int, int] | None]]:
        """Parse an ALTO XML file. Returns list of (bbox, text, page_size) - for each text entry."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        page = root.find(".//alto:Page", self.ALTO_NAMESPACE)

        # Page size metadata
        if page is not None:
            page_size = (int(page.get("WIDTH")), int(page.get("HEIGHT")))
        else:
            page_size = None

        # Collect all the positions and sizes of the available texts    
        samples = []
        for string in root.findall(".//alto:String", self.ALTO_NAMESPACE):
            text = string.get("CONTENT", "").strip()
            if not text:
                continue
            x = int(string.get("HPOS", 0))
            y = int(string.get("VPOS", 0))
            w = int(string.get("WIDTH", 0))
            h = int(string.get("HEIGHT", 0))
            samples.append(((x, y, x + w, y + h), text, page_size))
        return samples
    
    def parse_page(self, xml_path: Path) -> list[tuple[tuple, str, None]]:
        """Parse a PAGE XML file. Returns list of (bbox, text, None) per available text."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        samples = []

        # Each text appears on its own TextLine
        for line in root.findall(".//page:TextLine", self.PAGE_NAMESPACE):
            unicode_text = line.find("page:TextEquiv/page:Unicode", self.PAGE_NAMESPACE)
            if unicode_text is None or not (unicode_text.text or "").strip():
                continue
            text = unicode_text.text.strip()
            text_coords = line.find("page:Coords", self.PAGE_NAMESPACE)
            if text_coords is None:
                continue

            # The bounding box is a list of many jagged points, so we convert to a standard bbox
            bbox = self._bbox_from_points(text_coords.get("points", ""))
            samples.append((bbox, text, None))
        return samples
    
    ##### DATA ACCUMULATION FUNCTIONS #####
    def collect_samples(self, root_dir: str | Path, exclude: list[str] | None = None) -> list[dict]:
        """
        Walk the data directory and collect all the necessary data (XML + IMAGES) and its metadata
        """
        root_dir = Path(root_dir)
        exclude = set(exclude or [])
        samples = []

        # Walk the directory, skipping sources in EXCLUDE
        for dataset_dir in sorted(root_dir.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name in exclude:
                continue

            # Collect all (xml_root, img_root) pairs — handles both flat and one-level-deeper layouts (SENATSPROTOKOLLE)
            root_pairs: list[tuple[Path, Path]] = []
            if (dataset_dir / "xml").exists() and (dataset_dir / "images").exists():
                root_pairs.append((dataset_dir / "xml", dataset_dir / "images"))
            else:
                for sub in sorted(dataset_dir.iterdir()):
                    if sub.is_dir() and (sub / "xml").exists() and (sub / "images").exists():
                        root_pairs.append((sub / "xml", sub / "images"))

            for xml_root, img_root in root_pairs:
                for xml_path in sorted(xml_root.rglob("*.xml")):
                    rel = xml_path.relative_to(xml_root).parent
                    images_dir = img_root / rel

                    image_path = self._find_image(xml_path, images_dir)
                    if image_path is None:
                        continue

                    parser = self.parse_alto if self._is_alto(xml_path) else self.parse_page
                    try:
                        line_samples = parser(xml_path)
                    except ET.ParseError:
                        continue

                    for bbox, text, page_size in line_samples:
                        samples.append({
                            "image_path": image_path,
                            "bbox": bbox,
                            "page_size": page_size,
                            "text": text,
                            "dataset": dataset_dir.name,
                        })

        return samples
    
    ##### PRE-MODELING PREPERATION FUNCTIONS #####
    def build_vocab(self, samples: list[dict]) -> dict[str, int]:
        """Build a character-level vocabulary from sample texts."""
        chars = sorted({char for sample in samples for char in sample["text"]})
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, END_TOKEN: 2}
        for i, c in enumerate(chars, start=3):
            vocab[c] = i
        return vocab

    def image_transforms(self) -> transforms.Compose:
        """Defines the required transforms upon loading"""
        return transforms.Compose([
            transforms.Resize((self.IMG_HEIGHT, self.IMG_WIDTH)),
            transforms.ToTensor(),
        ])

    def _filter_samples(self, samples: list[dict]) -> list[dict]:
        """Remove samples whose image size doesn't match the ALTO page size. If we don't do this, cropping fails on the images as the XML is incorrect"""
        filtered = []
        for sample in samples:
            page_size = sample.get("page_size")
            if page_size is not None:
                with Image.open(sample["image_path"]) as img:
                    if (img.width, img.height) != page_size:
                        continue
            filtered.append(sample)
        return filtered
    
    def train_test_split(
        self,
        samples: list[dict], test_ratio: float = 0.15, val_ratio: float = 0.15, seed: int = 6500
    ) -> tuple[list[dict], list[dict], list[dict]]:
        
        gen = torch.Generator().manual_seed(seed)
        n_val = int(len(samples) * val_ratio)
        n_test = int(len(samples) * test_ratio)
        n_train = len(samples) - n_test - n_val
        indices = torch.randperm(len(samples), generator=gen).tolist()
        train = [samples[i] for i in indices[:n_train]]
        val = [samples[i] for i in  indices[n_train:n_train + n_val]]
        test = [samples[i] for i in indices[n_train + n_val:]]
        return train, val, test
    
    ##### HELPER FUNCTIONS #####

    def _bbox_from_points(self, points_str: str) -> tuple[int, int, int, int]:
        """Convert a PAGE XML polygon points string to (x_min, y_min, x_max, y_max)."""
        coords = [int(v) for pt in points_str.strip().split() for v in pt.split(",")]
        xs = coords[0::2]
        ys = coords[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    
    def _find_image(self, xml_path: Path, images_dir: Path) -> Path | None:
        """Finds the image path given an xml path. This works becuase we've set them up with the same naming conventions"""
        stem = xml_path.stem
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None
    
    def _is_alto(self, xml_path: Path) -> bool:
        """Given an XML file, determines if it is ALTO or PAGE based on the contents"""
        with open(xml_path, "r", encoding="utf-8") as f:
            for line in f:
                if "alto" in line.lower():
                    return True
                if "<Page" in line or "<PcGts" in line:
                    return False
        return False
    
    def _collate(self, batch: list[dict]) -> dict:
        """Custom data prep when returning from the DataLoader"""
        images = torch.stack([item["image"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])
        texts = [item["text"] for item in batch]
        datasets = [item["dataset"] for item in batch]
        return {"image": images, "target": targets, "text": texts, "dataset": datasets}
    

    ##### MAIN DATALOADER BUILD #####
    def build_dataloaders(
        self,
        root_dir: str | Path,
        exclude: list[str] | None = None,
        test_ratio: float = 0.15,
        val_ratio: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int], int]:
        """Build train and test DataLoaders.
        Returns (train_loader, val_loader, test_loader, vocab, max_len).
        """

        # Separate all the xml + img samples into train, val, test splits
        all_samples = self.collect_samples(root_dir, exclude=exclude)
        filtered_samples = self._filter_samples(all_samples)
        train_samples, val_samples, test_samples = self.train_test_split(filtered_samples, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)
        
        # Define the corpus metadata
        vocab = self.build_vocab(all_samples)
        max_len = max(len(s["text"]) for s in all_samples)

        # Define the datasets that get fed to the loaders
        train_ds = KurrentDataset(train_samples, vocab, max_len, transform=self.image_transforms())
        val_ds = KurrentDataset(val_samples, vocab, max_len, transform=self.image_transforms())
        test_ds = KurrentDataset(test_samples, vocab, max_len, transform=self.image_transforms())

        # Define the loaders
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=self._collate,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=self._collate
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=self._collate,
        )
        return train_loader, val_loader, test_loader, vocab, max_len

    def _collate_trocr(self, batch: list[dict]) -> dict:
        return {
            "image": [item["image"] for item in batch],
            "text": [item["text"] for item in batch],
            "dataset": [item["dataset"] for item in batch],
        }

    def build_trocr_dataloaders(
        self,
        root_dir: str | Path,
        exclude: list[str] | None = None,
        test_ratio: float = 0.15,
        val_ratio: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build train/val/test DataLoaders returning raw PIL crops for TrOCR.
        Returns (train_loader, val_loader, test_loader).
        """
        all_samples = self.collect_samples(root_dir, exclude=exclude)
        filtered_samples = self._filter_samples(all_samples)
        train_samples, val_samples, test_samples = self.train_test_split(
            filtered_samples, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed
        )

        train_ds = TrOCRDataset(train_samples)
        val_ds = TrOCRDataset(val_samples)
        test_ds = TrOCRDataset(test_samples)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=self._collate_trocr)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self._collate_trocr)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self._collate_trocr)
        return train_loader, val_loader, test_loader


if __name__ == "__main__":

    ROOT = "data/raw_data"
    SOURCES_TO_EXCLUDE = ["grandpa_letters_2", "DTA Handwritten OCR"]

    print("Building dataloaders...")
    train_loader, test_loader, vocab, max_len = build_dataloaders(
        ROOT, exclude=SOURCES_TO_EXCLUDE, batch_size=4, num_workers=0,
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test samples:  {len(test_loader.dataset)}")
    print(f"  Vocab size:    {len(vocab)}")
    print(f"  Max text len:  {max_len}")

    print("\nFirst batch:")
    batch = next(iter(train_loader))
    print(f"  images:  {batch['image'].shape}")
    print(f"  targets: {batch['target'].shape}")
    print(f"  texts:   {batch['text']}")
