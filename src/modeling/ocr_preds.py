import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.modeling.kurrent_ocr import TROCR_MODEL


@torch.no_grad()
def predict(model: VisionEncoderDecoderModel, processor: TrOCRProcessor, image_path: str | Path, device=None) -> str:
    if device is None:
        device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TrOCR text prediction on a single image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default=TROCR_MODEL, help="Model name or path")
    parser.add_argument("--processor", default="microsoft/trocr-base-handwritten", help="Processor name or path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(args.processor)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)

    text = predict(model, processor, args.image, device)
    print(text)
