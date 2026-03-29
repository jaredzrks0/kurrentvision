import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from src.modeling.dataset import DataloaderBuilder
from src.modeling.constants import (
    IMG_HEIGHT,
    IMG_WIDTH,
    PAD_TOKEN,
    UNK_TOKEN,
    END_TOKEN
    )


class LinearOCR(nn.Module):
    def __init__(self, vocab_size: int, max_len: int):
        super().__init__()
        input_dim = IMG_HEIGHT * IMG_WIDTH  # flattened B+W image
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
    for batch in tqdm(loader):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        # Push the images through the model, and compare with actuals for loss
        # To do this we convert logits to (Batch_size * max_len, vocab_size) and targets to (Batch_size * max_len)
        logits = model(images)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    average_token_loss = total_loss / len(loader.dataset)
    return average_token_loss


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
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=-1)
        mask = targets != pad_idx
        correct_chars += ((preds == targets) & mask).sum().item()
        total_chars += mask.sum().item()

    avg_loss = total_loss / len(loader.dataset)
    char_acc = correct_chars / total_chars if total_chars > 0 else 0.0
    return avg_loss, char_acc


def decode_predictions(preds: torch.Tensor, idx_to_char: dict[int, str]) -> list[str]:
    """Convert model output indices back to strings."""
    texts = []
    for seq in preds:
        chars = [idx_to_char.get(i.item(), "") for i in seq]
        text = "".join(chars).rstrip(idx_to_char.get(0, ""))
        texts.append(text)
    return texts


if __name__ == "__main__":
    ROOT = "data/raw_data"
    EXCLUDE = ["grandpa_letters_2"]
    EPOCHS = 5
    BATCH_SIZE = 32
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading data...")
    builder = DataloaderBuilder()
    train_loader, val_loader, test_loader, vocab, max_len = builder.build_dataloaders(
        ROOT, exclude=EXCLUDE, batch_size=BATCH_SIZE, num_workers=0,
    )
    print(f"  Train: {len(train_loader.dataset)}  Test: {len(test_loader.dataset)}")
    print(f"  Vocab: {len(vocab)}  Max len: {max_len}")

    model = LinearOCR(vocab_size=len(vocab), max_len=max_len + 1).to(device) # Add 1 to max_len bc everything now has END_TOKEN
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD_TOKEN])

    idx_to_char = {i: c for c, i in vocab.items()}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, vocab, device)
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  char_acc={test_acc:.4f}")

    # Show a few predictions
    batch = next(iter(test_loader))
    logits = model(batch["image"].to(device))
    preds = decode_predictions(logits.argmax(dim=-1).cpu(), idx_to_char)
    print("\nSample predictions:")
    for i in range(min(5, len(preds))):
        print(f"  true: {batch['text'][i]!r}")
        print(f"  pred: {preds[i]!r}")
        print()
