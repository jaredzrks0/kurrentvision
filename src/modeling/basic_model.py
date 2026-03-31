import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.modeling.datasets.basic_model import build_dataloaders, build_synthetic_dataloaders
from src.modeling.constants import (
    IMG_HEIGHT,
    IMG_WIDTH,
    PAD_TOKEN,
    UNK_TOKEN,
    END_TOKEN,
    BATCH_SIZE,
    EPOCHS,
    LR,

    )


class LinearOCR(nn.Module):
    def __init__(self, vocab_size: int, max_len: int):
        super().__init__()
        input_dim = 3 * IMG_HEIGHT * IMG_WIDTH  # flattened RGB image
        self.fc = nn.Linear(input_dim, vocab_size * max_len)
        self.vocab_size = vocab_size
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input X is of shape (Batch, Channels, Height, Width)"""
        batch_size = x.size(0)
        
        # Flatten to batch size, so now shape (batch_size, C*H*W)
        x = torch.flatten(x, 1)
        
        # Fully connected layer, from H*W --> vocab_size * max_len
        # This output size is to predict any vocab word max_len times
        x = self.fc(x)
        x = x.reshape(batch_size, self.max_len, self.vocab_size)
        return x


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    batch_grad_norms = []
    for batch in tqdm(loader):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        # Push the images through the model, and compare with actuals for loss
        # To do this we convert logits to (Batch_size * max_len, vocab_size) and targets to (Batch_size * max_len)
        logits = model(images)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()

        # Capture gradient norm before optimizer step
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        batch_grad_norms.append(total_norm ** 0.5)

        optimizer.step()
        total_loss += loss.item()

    average_token_loss = total_loss / len(loader.dataset)
    avg_grad_norm = sum(batch_grad_norms) / len(batch_grad_norms)
    return average_token_loss, avg_grad_norm


@torch.no_grad()
def evaluate(model, loader, criterion, vocab, device):
    model.eval()
    total_loss = 0.0
    correct_chars = 0
    total_chars = 0
    pad_idx = vocab[PAD_TOKEN]

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        logits = model(images)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        mask = targets != pad_idx
        correct_chars += ((preds == targets) & mask).sum().item()
        total_chars += mask.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    char_acc = correct_chars / total_chars if total_chars > 0 else 0.0
    return avg_loss, char_acc


def save_training_plots(history: dict, save_dir: Path) -> None:
    """Save training metric plots to disk."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], "o-", label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], "o-", label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Character Accuracy
    axes[0, 1].plot(epochs, history["val_char_acc"], "o-", label="Val", color="tab:orange")
    axes[0, 1].set_title("Character Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
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

    fig.suptitle("Basic Model Training Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "training_metrics.png", dpi=150)
    plt.close(fig)
    print(f"Plots saved to {save_dir / 'training_metrics.png'}")


def decode_predictions(preds: torch.Tensor, idx_to_char: dict[int, str]) -> list[str]:
    """Convert model output indices back to strings."""
    texts = []
    for seq in preds:
        chars = [idx_to_char.get(i.item(), "") for i in seq]
        text = "".join(chars).rstrip(idx_to_char.get(0, ""))
        texts.append(text)
    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LinearOCR model")
    parser.add_argument("--data", choices=["raw", "synthetic", "both"], default="raw",
                        help="Which dataset(s) to train on (default: raw)")
    parser.add_argument("--raw-dir", default="data/raw_data", help="Path to raw data directory")
    parser.add_argument("--synthetic-dir", default="data/synthetic_data", help="Path to synthetic data directory")
    parser.add_argument("--exclude", nargs="*", default=["grandpa_letters_2"], help="Raw data sources to exclude")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading data (mode={args.data})...")
    if args.data == "raw":
        train_loader, val_loader, test_loader, vocab, max_len = build_dataloaders(
            args.raw_dir, exclude=args.exclude, batch_size=BATCH_SIZE, num_workers=0,
        )
    elif args.data == "synthetic":
        train_loader, val_loader, test_loader, vocab, max_len = build_synthetic_dataloaders(
            args.synthetic_dir, batch_size=BATCH_SIZE, num_workers=0,
        )
    else:  # both
        train_loader, val_loader, test_loader, vocab, max_len = build_dataloaders(
            args.raw_dir, exclude=args.exclude, synthetic_dir=args.synthetic_dir,
            batch_size=BATCH_SIZE, num_workers=0,
        )
    print(f"  Train: {len(train_loader.dataset)}  Test: {len(test_loader.dataset)}")
    print(f"  Vocab: {len(vocab)}  Max len: {max_len}")

    model = LinearOCR(vocab_size=len(vocab), max_len=max_len + 1).to(device) # Add 1 to max_len bc everything now has END_TOKEN
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD_TOKEN])

    idx_to_char = {i: c for c, i in vocab.items()}

    history = {"train_loss": [], "val_loss": [], "val_char_acc": [], "grad_norm": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, grad_norm = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, vocab, device)
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  char_acc={val_acc:.4f}  grad_norm={grad_norm:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_char_acc"].append(val_acc)
        history["grad_norm"].append(grad_norm)

    # Show a few predictions
    batch = next(iter(test_loader))
    logits = model(batch["image"].to(device))
    preds = decode_predictions(logits.argmax(dim=-1).cpu(), idx_to_char)
    print("\nSample predictions:")
    for i in range(min(5, len(preds))):
        print(f"  true: {batch['text'][i]!r}")
        print(f"  pred: {preds[i]!r}")
        print()

    # Save model and plots
    save_dir = Path("models/basic_model")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_plots(history, save_dir)
    save_path = save_dir / f"linear_ocr_{args.data}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "max_len": max_len,
        "data_mode": args.data,
    }, save_path)
    print(f"Model saved to {save_path}")
