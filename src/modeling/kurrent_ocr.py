import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.modeling.dataset import DataloaderBuilder
from src.modeling.constants import EPOCHS, BATCH_SIZE


TROCR_MODEL = "dh-unibe/trocr-kurrent"
LR = 1e-5


def train_one_epoch(model, processor, loader, optimizer, device):
    model.train()
    total_loss = 0.0
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

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, processor, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader):
        pixel_values = processor(images=[to_pil_image(img) for img in batch["image"]], return_tensors="pt").pixel_values.to(device)
        labels = processor.tokenizer(
            batch["text"], return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)
        labels[labels == processor.tokenizer.pad_token_id] = -100

        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()

    return total_loss / len(loader.dataset)


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

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)

    print("Loading data...")
    builder = DataloaderBuilder()
    train_loader, val_loader, test_loader, vocab, max_len = builder.build_dataloaders(
        ROOT, exclude=EXCLUDE, batch_size=BATCH_SIZE, num_workers=0,
    )
    print(f"  Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, processor, train_loader, optimizer, device)
        val_loss = evaluate(model, processor, val_loader, device)
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    decode_predictions(model, processor, test_loader, device)
