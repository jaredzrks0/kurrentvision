"""
Fine-tune a Kraken segmentation model on Kurrent handwriting data.

Datasets and their XML formats:
  ALTO  - grandpa_letters, UGreifswald Senatsprotokolle
  PAGE  - DTA Handwritten OCR, Digitale Schriftkunde Archive Bayerns
"""

import re
from pathlib import Path

import torch
from lightning.pytorch import Callback
from kraken.train import KrakenTrainer, BLLASegmentationDataModule, BLLASegmentationModel
from kraken.configs import BLLASegmentationTrainingConfig, BLLASegmentationTrainingDataConfig

IMAGE_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

DATASETS = {
    "grandpa_letters": {"root": "data/raw_data/grandpa_letters", "fmt": "alto"},
    "senatsprotokolle": {"root": "data/raw_data/UGreifswald Senatsprotokolle", "fmt": "alto"},
    "dta": {"root": "data/raw_data/DTA Handwritten OCR", "fmt": "page"},
    "bayerns": {"root": "data/raw_data/Digitale Schriftkunde Archive Bayerns", "fmt": "page"},
}

_SKIP_FILENAMES = {"mets.xml", "metadata.xml", "mets_local.xml"}
_FILENAME_TAG_RE = re.compile(r"<fileName\s*>(.*?)</fileName>", re.IGNORECASE)
_DESCRIPTION_RE = re.compile(r"(<Description>)", re.IGNORECASE)


class EpochLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {k: f"{v:.4f}" for k, v in trainer.callback_metrics.items()}
        print(f"  Epoch {trainer.current_epoch + 1}: {metrics}")


def _find_image(xml_path: Path) -> Path | None:
    stem = xml_path.stem
    for candidate_dir in [xml_path.parent, xml_path.parent.parent / "images"]:
        for ext in IMAGE_EXTS:
            img = candidate_dir / (stem + ext)
            if img.exists():
                return img
    return None


def _patch_alto_xml(xml_path: Path, image_path: Path, patch_dir: Path) -> Path:
    text = xml_path.read_text(encoding="utf-8")
    abs_image = str(image_path.resolve())

    existing = _FILENAME_TAG_RE.search(text)
    if existing:
        current = existing.group(1).strip()
        # Check if the referenced file exists relative to the XML's directory
        resolved = (xml_path.parent / current).resolve()
        if resolved.exists():
            return xml_path  # Already valid, no patch needed
        # Replace the bad filename with the absolute path
        patched = _FILENAME_TAG_RE.sub(f"<fileName>{abs_image}</fileName>", text, count=1)
    else:
        injection = f"<sourceImageInformation><fileName>{abs_image}</fileName></sourceImageInformation>"
        patched = _DESCRIPTION_RE.sub(rf"\1{injection}", text, count=1)

    out_path = patch_dir / xml_path.relative_to(xml_path.anchor)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patched, encoding="utf-8")
    return out_path


def collect_xml_files(selected: dict, patch_dir: Path) -> list[str]:
    files = []
    for name, cfg in selected.items():
        root = Path(cfg["root"])
        if not root.exists():
            print(f"  [skip] {root} not found")
            continue
        xml_files = [f for f in root.rglob("*.xml") if f.name.lower() not in _SKIP_FILENAMES]
        resolved, skipped = [], 0
        for xml in xml_files:
            img = _find_image(xml)
            if img is None:
                skipped += 1
                continue
            patched = _patch_alto_xml(xml, img, patch_dir)
            resolved.append(str(patched))
        print(f"  {name}: {len(resolved)} usable XML files ({cfg['fmt']}), {skipped} skipped (no image)")
        files.extend(resolved)
    return files


def run(
    datasets: list[str] = ["all"],
    output_dir: str = "models/kraken",
    seg_model: str | None = None,
    epochs: int = 50,
    augment: bool = True,
):
    torch.set_float32_matmul_precision("high")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = output_dir / "patched_xml"
    patch_dir.mkdir(exist_ok=True)

    selected = DATASETS if "all" in datasets else {k: DATASETS[k] for k in datasets}

    print("\n=== Collecting training files ===")
    training_files = collect_xml_files(selected, patch_dir)

    if not training_files:
        print("No XML files found. Exiting.")
        return

    print(f"\nTotal: {len(training_files)} XML files")

    print("\n=== Fine-tuning segmentation model ===")
    dm_config = BLLASegmentationTrainingDataConfig(
        training_data=training_files,
        format_type="xml",
        augment=augment,
        partition=0.9,
        num_workers=0,
    )
    m_config = BLLASegmentationTrainingConfig(
        training_data=training_files,
        epochs=epochs,
        checkpoint_path=str(output_dir / "kurrent_seg"),
    )
    data_module = BLLASegmentationDataModule(dm_config)
    trainer = KrakenTrainer(
        max_epochs=epochs,
        enable_progress_bar=True,
        callbacks=[EpochLogger()],
    )

    with trainer.init_module():
        if seg_model:
            model = BLLASegmentationModel.load_from_weights(seg_model, config=m_config)
        else:
            model = BLLASegmentationModel(m_config)

    trainer.fit(model, data_module)

    print(f"\nDone. Model saved to {output_dir}/")
