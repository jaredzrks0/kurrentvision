from pathlib import Path
from xml.etree import ElementTree as ET

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from src.modeling.constants import IMG_WIDTH, IMG_HEIGHT

ALTO_NAMESPACE = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
PAGE_NAMESPACE = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


# ── XML Parsers ──────────────────────────────────────────────────────────────

def parse_alto(xml_path: Path) -> list[tuple[tuple, str, tuple[int, int] | None]]:
    """Parse an ALTO XML file. Returns list of (bbox, text, page_size) - for each text entry."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    page = root.find(".//alto:Page", ALTO_NAMESPACE)

    if page is not None:
        page_size = (int(page.get("WIDTH")), int(page.get("HEIGHT")))
    else:
        page_size = None

    samples = []
    for string in root.findall(".//alto:String", ALTO_NAMESPACE):
        text = string.get("CONTENT", "").strip()
        if not text:
            continue
        x = int(string.get("HPOS", 0))
        y = int(string.get("VPOS", 0))
        w = int(string.get("WIDTH", 0))
        h = int(string.get("HEIGHT", 0))
        samples.append(((x, y, x + w, y + h), text, page_size))
    return samples


def parse_page(xml_path: Path) -> list[tuple[tuple, str, None]]:
    """Parse a PAGE XML file. Returns list of (bbox, text, None) per available text."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    samples = []

    for line in root.findall(".//page:TextLine", PAGE_NAMESPACE):
        unicode_text = line.find("page:TextEquiv/page:Unicode", PAGE_NAMESPACE)
        if unicode_text is None or not (unicode_text.text or "").strip():
            continue
        text = unicode_text.text.strip()
        text_coords = line.find("page:Coords", PAGE_NAMESPACE)
        if text_coords is None:
            continue

        bbox = _bbox_from_points(text_coords.get("points", ""))
        samples.append((bbox, text, None))
    return samples


# ── Sample Collection ────────────────────────────────────────────────────────

def collect_raw_samples(root_dir: str | Path, exclude: list[str] | None = None) -> list[dict]:
    """Walk the data directory and collect all XML + image sample metadata."""
    root_dir = Path(root_dir)
    exclude = set(exclude or [])
    samples = []

    for dataset_dir in sorted(root_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in exclude:
            continue

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

                image_path = _find_image(xml_path, images_dir)
                if image_path is None:
                    continue

                parser = parse_alto if _is_alto(xml_path) else parse_page
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


def collect_synthetic_samples(synthetic_dir: str | Path) -> list[dict]:
    """Load paired image/text files from the synthetic data directory."""
    synthetic_dir = Path(synthetic_dir)
    images_dir = synthetic_dir / "images"
    texts_dir = synthetic_dir / "texts"
    samples = []
    for img_path in sorted(images_dir.glob("*.png")):
        txt_path = texts_dir / img_path.with_suffix(".txt").name
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        samples.append({"image_path": img_path, "text": text, "dataset": "synthetic"})
    return samples


# ── Vocab / Transforms / Splitting ──────────────────────────────────────────

def build_vocab(samples: list[dict]) -> dict[str, int]:
    """Build a character-level vocabulary from sample texts."""
    from src.modeling.constants import PAD_TOKEN, UNK_TOKEN, END_TOKEN
    chars = sorted({char for sample in samples for char in sample["text"]})
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, END_TOKEN: 2}
    for i, c in enumerate(chars, start=3):
        vocab[c] = i
    return vocab


def image_transforms() -> transforms.Compose:
    """Resize and convert to tensor."""
    return transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])


def filter_raw_samples(samples: list[dict]) -> list[dict]:
    """Remove samples whose image size doesn't match the ALTO page size."""
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
    samples: list[dict], test_ratio: float = 0.15, val_ratio: float = 0.15, seed: int = 6500
) -> tuple[list[dict], list[dict], list[dict]]:
    gen = torch.Generator().manual_seed(seed)
    n_val = int(len(samples) * val_ratio)
    n_test = int(len(samples) * test_ratio)
    n_train = len(samples) - n_test - n_val
    indices = torch.randperm(len(samples), generator=gen).tolist()
    train = [samples[i] for i in indices[:n_train]]
    val = [samples[i] for i in indices[n_train:n_train + n_val]]
    test = [samples[i] for i in indices[n_train + n_val:]]
    return train, val, test


# ── Private Helpers ──────────────────────────────────────────────────────────

def _bbox_from_points(points_str: str) -> tuple[int, int, int, int]:
    """Convert a PAGE XML polygon points string to (x_min, y_min, x_max, y_max)."""
    coords = [int(v) for pt in points_str.strip().split() for v in pt.split(",")]
    xs = coords[0::2]
    ys = coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _find_image(xml_path: Path, images_dir: Path) -> Path | None:
    """Finds the image path given an xml path."""
    stem = xml_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def _is_alto(xml_path: Path) -> bool:
    """Given an XML file, determines if it is ALTO or PAGE based on the contents."""
    with open(xml_path, "r", encoding="utf-8") as f:
        for line in f:
            if "alto" in line.lower():
                return True
            if "<Page" in line or "<PcGts" in line:
                return False
    return False
