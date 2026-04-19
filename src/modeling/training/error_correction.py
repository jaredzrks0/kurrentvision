import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from modeling.datasets.trocr import build_dataloaders
from modeling.constants import BATCH_SIZE, LR
from modeling.utils import save_error_correction_plots, cer


CORRECTOR_MODEL = "MRNH/mbart-german-grammar-corrector"
TROCR_MODEL = "dh-unibe/trocr-kurrent"
MAX_LEN = 128

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def correct(texts: list[str], model, tokenizer, device) -> list[str]:
    """Run the grammar corrector on a list of OCR strings."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
    generated = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"],
        max_length=MAX_LEN,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


@torch.no_grad()
def _ocr_predictions(ocr_model, ocr_processor, images, device) -> list[str]:
    pixel_values = ocr_processor(images=images, return_tensors="pt").pixel_values.to(device)
    generated_ids = ocr_model.generate(pixel_values)
    return ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)


def _corrector_loss(noisy_texts, clean_texts, model, tokenizer, device):
    inputs = tokenizer(
        noisy_texts,
        text_target=clean_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(device)
    labels = inputs["labels"]
    labels[labels == tokenizer.pad_token_id] = -100
    h = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=labels,
    )
    return h.loss


def train_one_epoch(model, tokenizer, ocr_model, ocr_processor, loader, optimizer, device, compute_cer=False):
    model.train()
    ocr_model.eval()

    total_loss = 0.0
    all_ocr, all_corrected, all_targets = [], [], []
    batch_grad_norms = []

    for batch in tqdm(loader):
        ocr_preds = _ocr_predictions(ocr_model, ocr_processor, batch["image"], device)
        ground_truths = batch["text"]

        loss = _corrector_loss(ocr_preds, ground_truths, model, tokenizer, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        batch_grad_norms.append(total_norm)
        optimizer.step()

        total_loss += loss.item()

        if compute_cer:
            with torch.no_grad():
                corrected = correct(ocr_preds, model, tokenizer, device)
            all_ocr.extend(ocr_preds)
            all_corrected.extend(corrected)
            all_targets.extend(ground_truths)

    avg_loss = total_loss / len(loader.dataset)
    ocr_cer = cer(all_ocr, all_targets) if compute_cer else None
    corrected_cer = cer(all_corrected, all_targets) if compute_cer else None
    avg_grad_norm = sum(batch_grad_norms) / len(batch_grad_norms)
    return avg_loss, ocr_cer, corrected_cer, avg_grad_norm


@torch.no_grad()
def evaluate(model, tokenizer, ocr_model, ocr_processor, loader, device, compute_cer=False):
    model.eval()
    ocr_model.eval()

    total_loss = 0.0
    all_ocr, all_corrected, all_targets = [], [], []

    for batch in tqdm(loader):
        noisy = _ocr_predictions(ocr_model, ocr_processor, batch["image"], device)
        clean = batch["text"]

        loss = _corrector_loss(noisy, clean, model, tokenizer, device)
        total_loss += loss.item()

        if compute_cer:
            corrected = correct(noisy, model, tokenizer, device)
            all_ocr.extend(noisy)
            all_corrected.extend(corrected)
            all_targets.extend(clean)

    avg_loss = total_loss / len(loader.dataset)
    ocr_cer = cer(all_ocr, all_targets) if compute_cer else None
    corrected_cer = cer(all_corrected, all_targets) if compute_cer else None
    return avg_loss, ocr_cer, corrected_cer


def _freeze_backbone(model, unfrozen_decoder_layers=2):
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    decoder_layers = model.model.decoder.layers
    for layer in decoder_layers[:-unfrozen_decoder_layers]:
        for param in layer.parameters():
            param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable params: %s / %s (%.1f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)


@torch.no_grad()
def sample_predictions(model, tokenizer, ocr_model, ocr_processor, loader, device, n=5):
    model.eval()
    ocr_model.eval()
    batch = next(iter(loader))
    noisy = _ocr_predictions(ocr_model, ocr_processor, batch["image"], device)
    corrected = correct(noisy, model, tokenizer, device)
    print("\nSample predictions:")
    for i in range(min(n, len(corrected))):
        print(f"true: {batch['text'][i]}")
        print(f"ocr: {noisy[i]}")
        print(f"corrected: {corrected[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune grammar corrector on OCR output")
    parser.add_argument("--data", choices=["raw", "synthetic", "both"], default="raw")
    parser.add_argument("--raw-dir", default="data/raw_data")
    parser.add_argument("--synthetic-dir", default="data/synthetic_data")
    parser.add_argument("--exclude", nargs="*", default=["grandpa_letters", "grandpa_letters_2"])
    parser.add_argument("--ocr-model", default=TROCR_MODEL, help="Path or HF ID of the TrOCR model to use for generating OCR inputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--compute-cer", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Device: %s", device)

    ocr_model_path = Path(args.ocr_model)
    if ocr_model_path.exists():
        ocr_model_id = str(ocr_model_path.resolve())
        logger.info("OCR model: local checkpoint at %s", ocr_model_id)
    else:
        ocr_model_id = args.ocr_model
        logger.info("OCR model: HuggingFace '%s'", ocr_model_id)

    logger.info("Loading TrOCR processor from HuggingFace: microsoft/trocr-base-handwritten")
    ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    logger.info("Loading TrOCR model from: %s", ocr_model_id)
    ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_id).to(device)
    for p in ocr_model.parameters():
        p.requires_grad = False

    logger.info("Loading grammar corrector tokenizer from HuggingFace: %s", CORRECTOR_MODEL)
    tokenizer = MBart50TokenizerFast.from_pretrained(CORRECTOR_MODEL, src_lang="de_DE", tgt_lang="de_DE")
    logger.info("Loading grammar corrector model from HuggingFace: %s", CORRECTOR_MODEL)
    corrector = MBartForConditionalGeneration.from_pretrained(CORRECTOR_MODEL).to(device)
    _freeze_backbone(corrector, unfrozen_decoder_layers=2)

    if args.data == "raw":
        train_loader, val_loader, test_loader = build_dataloaders(
            args.raw_dir, exclude=args.exclude, batch_size=BATCH_SIZE, num_workers=0,
        )
    elif args.data == "synthetic":
        train_loader, val_loader, test_loader = build_dataloaders(
            synthetic_dir=args.synthetic_dir, batch_size=BATCH_SIZE, num_workers=0,
        )
    else:
        train_loader, val_loader, test_loader = build_dataloaders(
            args.raw_dir, exclude=args.exclude, synthetic_dir=args.synthetic_dir,
            batch_size=BATCH_SIZE, num_workers=0,
        )
    logger.info("Train: %d  Val: %d  Test: %d", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    optimizer = torch.optim.Adam(corrector.parameters(), lr=LR)

    logger.info("=" * 50)
    logger.info("Corrector model: %s", CORRECTOR_MODEL)
    logger.info("OCR model: %s", ocr_model_id)
    logger.info("Epochs: %d", args.epochs)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("Learning rate: %s", LR)
    logger.info("Device: %s", device)
    logger.info("Architecture:\n%s", corrector)
    logger.info("=" * 50)

    history = {
        "train_loss": [], "val_loss": [],
        "train_ocr_cer": [], "train_corrected_cer": [],
        "val_ocr_cer": [], "val_corrected_cer": [],
        "grad_norm": [],
    }

    save_dir = Path("models/grammar_corrector")
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ocr_cer, train_corrected_cer, grad_norm = train_one_epoch(
            corrector, tokenizer, ocr_model, ocr_processor,
            train_loader, optimizer, device, compute_cer=args.compute_cer,
        )
        val_loss, val_ocr_cer, val_corrected_cer = evaluate(
            corrector, tokenizer, ocr_model, ocr_processor,
            val_loader, device, compute_cer=args.compute_cer,
        )
        if args.compute_cer:
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  grad_norm=%.4f  "
                "train_cer: ocr=%.4f corrected=%.4f  val_cer: ocr=%.4f corrected=%.4f",
                epoch, args.epochs, train_loss, val_loss, grad_norm,
                train_ocr_cer, train_corrected_cer, val_ocr_cer, val_corrected_cer,
            )
        else:
            logger.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  grad_norm=%.4f", epoch, args.epochs, train_loss, val_loss, grad_norm)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ocr_cer"].append(train_ocr_cer)
        history["train_corrected_cer"].append(train_corrected_cer)
        history["val_ocr_cer"].append(val_ocr_cer)
        history["val_corrected_cer"].append(val_corrected_cer)
        history["grad_norm"].append(grad_norm)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            corrector.generation_config.output_hidden_states = False
            corrector.save_pretrained(save_dir / f"corrector_{args.data}_best")
            tokenizer.save_pretrained(save_dir / f"corrector_{args.data}_best")
            logger.info("New best model saved (val_loss=%.4f)", best_val_loss)

    sample_predictions(corrector, tokenizer, ocr_model, ocr_processor, val_loader, device)

    save_error_correction_plots(history, save_dir)
    corrector.generation_config.output_hidden_states = False
    corrector.save_pretrained(save_dir / f"corrector_{args.data}_final")
    tokenizer.save_pretrained(save_dir / f"corrector_{args.data}_final")
    logger.info("Final model saved to %s", save_dir / f"corrector_{args.data}_final")
