import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.modeling.datasets.trocr import build_dataloaders, build_synthetic_dataloaders
from src.modeling.constants import EPOCHS, BATCH_SIZE


TROCR_MODEL = "dh-unibe/trocr-kurrent"
LR = 1e-5


def _edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
    return dp[-1]


def _cer(preds: list[str], targets: list[str]) -> float:
    total_edits = sum(_edit_distance(pred, true) for pred, true in zip(preds, targets))
    total_chars = sum(len(true) for true in targets)
    return total_edits / total_chars if total_chars > 0 else 0.0


def train_one_epoch(model, processor, loader, optimizer, device, compute_char_acc: bool = False):
    model.train()
    total_loss = 0.0
    batch_grad_norms = []
    all_preds, all_targets = [], []
    for batch in tqdm(loader):
        pixel_values = processor(images=batch["image"], return_tensors="pt").pixel_values.to(device)
        labels = processor.tokenizer(
            batch["text"], return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)
        labels[labels == processor.tokenizer.pad_token_id] = -100

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        batch_grad_norms.append(total_norm ** 0.5)

        optimizer.step()
        total_loss += loss.item()

        if compute_char_acc:
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            all_preds.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
            all_targets.extend(batch["text"])

    avg_loss = total_loss / len(loader.dataset)
    cer = _cer(all_preds, all_targets) if compute_char_acc else None
    avg_grad_norm = sum(batch_grad_norms) / len(batch_grad_norms)
    return avg_loss, cer, avg_grad_norm


@torch.no_grad()
def evaluate(model, processor, loader, device, compute_char_acc: bool = False):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in tqdm(loader):
        pixel_values = processor(images=batch["image"], return_tensors="pt").pixel_values.to(device)
        labels = processor.tokenizer(
            batch["text"], return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)
        labels[labels == processor.tokenizer.pad_token_id] = -100

        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()

        if compute_char_acc:
            generated_ids = model.generate(pixel_values)
            all_preds.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
            all_targets.extend(batch["text"])

    avg_loss = total_loss / len(loader.dataset)
    char_acc = _cer(all_preds, all_targets) if compute_char_acc else None
    return avg_loss, char_acc


@torch.no_grad()
def decode_predictions(model, processor, loader, device, n=5) -> None:
    model.eval()
    batch = next(iter(loader))
    pixel_values = processor(images=batch["image"], return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print("\nSample predictions:")
    for i in range(min(n, len(preds))):
        print(f"  true: {batch['text'][i]!r}")
        print(f"  pred: {preds[i]!r}")
        print()


def save_training_plots(history: dict, save_dir: Path) -> None:
    """Save training metric plots to disk."""
    epochs = range(1, len(history["train_loss"]) + 1)
    has_cer = history["train_cer"][0] is not None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], "o-", label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], "o-", label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # CER
    if has_cer:
        axes[0, 1].plot(epochs, history["train_cer"], "o-", label="Train")
        axes[0, 1].plot(epochs, history["val_cer"], "o-", label="Val")
        axes[0, 1].set_title("Character Error Rate")
        axes[0, 1].set_ylabel("CER (lower is better)")
    else:
        axes[0, 1].text(0.5, 0.5, "CER not computed\n(use --compute-cer)",
                        ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].set_title("Character Error Rate")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Gradient Norm
    axes[1, 0].plot(epochs, history["grad_norm"], "o-", color="tab:red")
    axes[1, 0].set_title("Average Gradient Norm")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("L2 Norm")
    axes[1, 0].grid(True)

    # Train vs Val gap (overfitting indicator)
    gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
    axes[1, 1].plot(epochs, gap, "o-", color="tab:purple")
    axes[1, 1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1, 1].set_title("Generalization Gap (Val - Train Loss)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss Difference")
    axes[1, 1].grid(True)

    fig.suptitle("TrOCR Training Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "training_metrics.png", dpi=150)
    plt.close(fig)
    print(f"Plots saved to {save_dir / 'training_metrics.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on Kurrent data")
    parser.add_argument("--data", choices=["raw", "synthetic", "both"], default="raw",
                        help="Which dataset(s) to train on (default: raw)")
    parser.add_argument("--raw-dir", default="data/raw_data", help="Path to raw data directory")
    parser.add_argument("--synthetic-dir", default="data/synthetic_data", help="Path to synthetic data directory")
    parser.add_argument("--exclude", nargs="*", default=["grandpa_letters_2"], help="Raw data sources to exclude")
    parser.add_argument("--compute-cer", action="store_true", default=True,
                        help="Compute character error rate each epoch (slower)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)

    print(f"Loading data (mode={args.data})...")
    if args.data == "raw":
        train_loader, val_loader, test_loader = build_dataloaders(
            args.raw_dir, exclude=args.exclude, batch_size=BATCH_SIZE, num_workers=0,
        )
    elif args.data == "synthetic":
        train_loader, val_loader, test_loader = build_synthetic_dataloaders(
            args.synthetic_dir, batch_size=BATCH_SIZE, num_workers=0,
        )
    else:  # both
        train_loader, val_loader, test_loader = build_dataloaders(
            args.raw_dir, exclude=args.exclude, synthetic_dir=args.synthetic_dir,
            batch_size=BATCH_SIZE, num_workers=0,
        )
    print(f"  Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "train_cer": [], "val_cer": [], "grad_norm": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_cer, grad_norm = train_one_epoch(model, processor, train_loader, optimizer, device, compute_char_acc=args.compute_cer)
        val_loss, val_cer = evaluate(model, processor, val_loader, device, compute_char_acc=args.compute_cer)
        cer_str = f"  train_cer={train_cer:.4f}  val_cer={val_cer:.4f}" if args.compute_cer else ""
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{cer_str}  grad_norm={grad_norm:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_cer"].append(train_cer)
        history["val_cer"].append(val_cer)
        history["grad_norm"].append(grad_norm)

    decode_predictions(model, processor, test_loader, device)

    # Save model and plots
    save_dir = Path("models/kurrent_ocr")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_plots(history, save_dir)
    save_path = save_dir / f"trocr_{args.data}"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
