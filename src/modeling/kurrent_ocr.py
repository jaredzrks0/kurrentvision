import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.modeling.dataset import DataloaderBuilder
from src.modeling.constants import EPOCHS, BATCH_SIZE


TROCR_MODEL = "dh-unibe/trocr-kurrent"
LR = 1e-5


def _char_accuracy(preds: list[str], targets: list[str]) -> float:
    correct = sum(p == t for pred, true in zip(preds, targets) for p, t in zip(pred, true))
    total = sum(len(true) for true in targets)
    return correct / total if total > 0 else 0.0


def train_one_epoch(model, processor, loader, optimizer, device, compute_char_acc: bool = False):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in tqdm(loader):
        pixel_values = processor(images=[to_pil_image(img) for img in batch["image"]], return_tensors="pt").pixel_values.to(device)
        labels = processor.tokenizer(
            batch["text"], return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)
        labels[labels == processor.tokenizer.pad_token_id] = -100

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if compute_char_acc:
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            all_preds.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
            all_targets.extend(batch["text"])

    avg_loss = total_loss / len(loader.dataset)
    char_acc = _char_accuracy(all_preds, all_targets) if compute_char_acc else None
    return avg_loss, char_acc


@torch.no_grad()
def evaluate(model, processor, loader, device, compute_char_acc: bool = False):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in tqdm(loader):
        pixel_values = processor(images=[to_pil_image(img) for img in batch["image"]], return_tensors="pt").pixel_values.to(device)
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
    char_acc = _char_accuracy(all_preds, all_targets) if compute_char_acc else None
    return avg_loss, char_acc


@torch.no_grad()
def decode_predictions(model, processor, loader, device, n=5) -> None:
    model.eval()
    batch = next(iter(loader))
    pixel_values = processor(images=[to_pil_image(img) for img in batch["image"]], return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print("\nSample predictions:")
    for i in range(min(n, len(preds))):
        print(f"  true: {batch['text'][i]!r}")
        print(f"  pred: {preds[i]!r}")
        print()


if __name__ == "__main__":
    ROOT = "data/raw_data"
    EXCLUDE = ["grandpa_letters_2"]
    COMPUTE_CHAR_ACC = True  # set True to compute char accuracy each epoch (slower)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)

    print("Loading data...")
    builder = DataloaderBuilder()
    train_loader, val_loader, test_loader, vocab, max_len = builder.build_dataloaders(
        ROOT, exclude=EXCLUDE, batch_size=BATCH_SIZE, num_workers=0,
    )
    print(f"  Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, processor, train_loader, optimizer, device, compute_char_acc=COMPUTE_CHAR_ACC)
        val_loss, val_acc = evaluate(model, processor, val_loader, device, compute_char_acc=COMPUTE_CHAR_ACC)
        acc_str = f"  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}" if COMPUTE_CHAR_ACC else ""
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{acc_str}")

    decode_predictions(model, processor, test_loader, device)
