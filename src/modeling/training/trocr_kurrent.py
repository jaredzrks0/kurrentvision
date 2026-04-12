import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from modeling.datasets.trocr import build_dataloaders
from modeling.constants import EPOCHS, BATCH_SIZE
from modeling.utils import save_training_plots, cer


TROCR_MODEL = "dh-unibe/trocr-kurrent"


def train_one_epoch(model, processor, loader, optimizer, device, compute_char_acc: bool = False):
    model.train()
    total_loss, all_preds, all_targets, avg_grad_norm = _forward_pass(
        model, processor, loader, device, compute_char_acc, is_eval=False, optimizer=optimizer
    )
    avg_loss = total_loss / len(loader.dataset)
    error_rate = cer(all_preds, all_targets) if compute_char_acc else None
    return avg_loss, error_rate, avg_grad_norm


@torch.no_grad()
def evaluate(model, processor, loader, device, compute_char_acc: bool = False):
    model.eval()
    total_loss, all_preds, all_targets, _ = _forward_pass(
        model, processor, loader, device, compute_char_acc, is_eval=True
    )
    avg_loss = total_loss / len(loader.dataset)
    char_acc = cer(all_preds, all_targets) if compute_char_acc else None
    return avg_loss, char_acc

def _forward_pass(model, processor, loader, device, compute_char_acc, is_eval=True, optimizer=None):
    all_preds, all_targets = [], []
    total_loss = 0.0
    batch_grad_norms = []
    for batch in tqdm(loader):
        pixel_values = processor(images=batch["image"], return_tensors="pt").pixel_values.to(device)
        labels = processor.tokenizer(
            batch["text"], return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)
        labels[labels == processor.tokenizer.pad_token_id] = -100

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        if not is_eval:
            _update_and_store_gradients(model=model, batch_grad_norms=batch_grad_norms, optimizer=optimizer, loss=loss)

        total_loss += loss.item()

        if compute_char_acc:
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            all_preds.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
            all_targets.extend(batch["text"])

    avg_grad_norm = sum(batch_grad_norms) / len(batch_grad_norms) if batch_grad_norms else None
    return total_loss, all_preds, all_targets, avg_grad_norm

def _update_and_store_gradients(model, batch_grad_norms,optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    batch_grad_norms.append(total_norm ** 0.5)
    optimizer.step()


@torch.no_grad()
def sample_predictions(model, processor, loader, device, n=5) -> None:
    model.eval()
    batch = next(iter(loader))
    pixel_values = processor(images=batch["image"], return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print("\nSample predictions:")
    for i in range(min(n, len(preds))):
        print(f"true: {batch['text'][i]}")
        print(f"pred: {preds[i]}")
        print()


def _freeze_backbone(model, unfrozen_decoder_layers=2):
    for param in model.encoder.parameters():
        param.requires_grad = False

    decoder_layers = model.decoder.model.decoder.layers
    for layer in decoder_layers[:-unfrozen_decoder_layers]:
        for param in layer.parameters():
            param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on Kurrent data")
    parser.add_argument("--data", choices=["raw", "synthetic", "both"], default="raw",
                        help="Which dataset(s) to train on (default: raw)")
    parser.add_argument("--raw-dir", default="data/raw_data", help="Path to raw data directory")
    parser.add_argument("--synthetic-dir", default="data/synthetic_data", help="Path to synthetic data directory")
    parser.add_argument("--exclude", nargs="*", default=["grandpa_letters", "grandpa_letters_2"], help="Raw data sources to exclude")
    parser.add_argument("--compute-cer", action="store_true", default=True,
                        help="Compute character error rate each epoch (slower)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Download both the model and its accompanying processor
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)
    _freeze_backbone(model)

    print(f"Loading data from {args.data})...")
    if args.data == "raw":
        train_loader, val_loader, test_loader = build_dataloaders(
            args.raw_dir, exclude=args.exclude, batch_size=BATCH_SIZE, num_workers=0,
        )
    elif args.data == "synthetic":
        train_loader, val_loader, test_loader = build_dataloaders(
            synthetic_dir=args.synthetic_dir, batch_size=BATCH_SIZE, num_workers=0,
        )
    else:  # both
        train_loader, val_loader, test_loader = build_dataloaders(
            args.raw_dir, exclude=args.exclude, synthetic_dir=args.synthetic_dir,
            batch_size=BATCH_SIZE, num_workers=0,
        )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=.00001)

    history = {"train_loss": [], "val_loss": [], "train_cer": [], "val_cer": [], "grad_norm": []}

    # Actually train the thing
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_cer, grad_norm = train_one_epoch(model, processor, train_loader, optimizer, device, compute_char_acc=args.compute_cer)
        val_loss, val_cer = evaluate(model, processor, val_loader, device, compute_char_acc=args.compute_cer)
        cer_str = f"train_cer={train_cer:.4f}  val_cer={val_cer:.4f}" if args.compute_cer else ""
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{cer_str}  grad_norm={grad_norm:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_cer"].append(train_cer)
        history["val_cer"].append(val_cer)
        history["grad_norm"].append(grad_norm)

    # Sample some validation texts to preview
    sample_predictions(model, processor, val_loader, device)

    # Save model and plots
    save_dir = Path("models/kurrent_ocr")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_plots(history, save_dir)
    save_path = save_dir / f"trocr_{args.data}"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
