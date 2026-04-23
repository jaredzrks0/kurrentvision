import argparse
import torch

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, MBartForConditionalGeneration, MBart50TokenizerFast

from modeling.datasets.base import parse_alto, parse_page, _is_alto
from modeling.utils import cer


class XmlInference:
    def __init__(self, ocr_model, ocr_processor, device, corrector=None, tokenizer=None, max_len: int = 128):
        self.ocr_model = ocr_model
        self.ocr_processor = ocr_processor
        self.corrector = corrector
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

        self.ocr_model.eval()
        if self.corrector is not None:
            self.corrector.eval()

    @torch.no_grad()
    def _ocr(self, crops: list) -> list[str]:
        pixel_values = self.ocr_processor(images=crops, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.ocr_model.generate(pixel_values)
        return self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)

    @torch.no_grad()
    def _correct(self, texts: list[str]) -> list[str]:
        if self.corrector is None:
            return texts
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len
        ).to(self.device)
        generated = self.corrector.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            forced_bos_token_id=self.tokenizer.lang_code_to_id["de_DE"],
            max_length=self.max_len,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def full_inference_from_path(
        self,
        xml_path: str | Path,
        image_path: str | Path,
        truth: str | None = None,
    ) -> dict:
        xml_path = Path(xml_path)
        image_path = Path(image_path)

        parser = parse_alto if _is_alto(xml_path) else parse_page
        line_samples = parser(xml_path)

        image = Image.open(image_path).convert("RGB")
        image.load()

        crops, xml_truths, bboxes = [], [], []
        for bbox, xml_text, _ in line_samples:
            crops.append(image.crop(bbox).copy())
            xml_truths.append(xml_text)
            bboxes.append(bbox)

        ocr_preds = self._ocr(crops)
        corrected_preds = self._correct(ocr_preds)

        lines = [
            {"bbox": bbox, "xml_truth": gt, "ocr": ocr, "corrected": corr}
            for bbox, gt, ocr, corr in zip(bboxes, xml_truths, ocr_preds, corrected_preds)
        ]

        full_ocr = " ".join(ocr_preds)
        full_corrected = " ".join(corrected_preds)
        ground_truth = truth if truth is not None else " ".join(xml_truths)

        return {
            "lines": lines,
            "full_text_ocr": full_ocr,
            "full_text_corrected": full_corrected,
            "ground_truth": ground_truth,
            "cer_ocr": cer([full_ocr], [ground_truth]),
            "cer_corrected": cer([full_corrected], [ground_truth]),
        }

    def full_inference_from_dataset(self, dataset: Dataset, batch_size: int = 16) -> dict:
        def collate(batch):
            return {
                "image": [x["image"] for x in batch],
                "text": [x["text"] for x in batch],
            }

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

        all_ocr, all_corrected, all_truths = [], [], []
        for batch in loader:
            ocr_preds = self._ocr(batch["image"])
            corrected_preds = self._correct(ocr_preds)
            all_ocr.extend(ocr_preds)
            all_corrected.extend(corrected_preds)
            all_truths.extend(batch["text"])

        samples = [
            {"truth": t, "ocr": o, "corrected": c}
            for t, o, c in zip(all_truths, all_ocr, all_corrected)
        ]

        return {
            "samples": samples,
            "cer_ocr": cer(all_ocr, all_truths),
            "cer_corrected": cer(all_corrected, all_truths),
        }


if __name__ == "__main__":
    from modeling.datasets.trocr import build_dataloaders

    parser = argparse.ArgumentParser(description="Run full OCR + error correction inference")
    parser.add_argument("--mode", choices=["path", "dataset"], default="path")
    parser.add_argument("--ocr-model", required=True, help="Path to local TrOCR checkpoint directory")
    parser.add_argument("--corrector-model", default=None, help="Path to local grammar corrector checkpoint directory (optional)")
    # path mode
    parser.add_argument("--xml", help="Path to the XML annotation file")
    parser.add_argument("--image", help="Path to the corresponding image file")
    # dataset mode
    parser.add_argument("--data", choices=["raw", "synthetic", "both"], default="both")
    parser.add_argument("--raw-dir", default="data/raw_data")
    parser.add_argument("--synthetic-dir", default="data/synthetic_data")
    parser.add_argument("--exclude", nargs="*", default=["grandpa_letters", "grandpa_letters_2"])
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--save-text", action="store_true", help="Save results to a text file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    ocr_model = VisionEncoderDecoderModel.from_pretrained(args.ocr_model).to(device)

    corrector, tokenizer = None, None
    if args.corrector_model:
        tokenizer = MBart50TokenizerFast.from_pretrained(args.corrector_model, src_lang="de_DE", tgt_lang="de_DE")
        corrector = MBartForConditionalGeneration.from_pretrained(args.corrector_model).to(device)

    inferencer = XmlInference(ocr_model, ocr_processor, device, corrector=corrector, tokenizer=tokenizer)

    if args.mode == "path":
        results = inferencer.full_inference_from_path(args.xml, args.image)
        print(f"\nCER (OCR): {results['cer_ocr']:.4f}")
        print(f"CER (corrected): {results['cer_corrected']:.4f}")
        print(f"\nFull OCR text:\n{results['full_text_ocr']}")
        print(f"\nFull corrected text:\n{results['full_text_corrected']}")
        print(f"\nGround truth:\n{results['ground_truth']}")
        print("\nPer-line breakdown:")
        for line in results["lines"]:
            print(f"  truth: {line['xml_truth']}")
            print(f"  ocr: {line['ocr']}")
            print(f"  corrected: {line['corrected']}")
            print()

        if args.save_text:
            out_path = Path(args.xml).with_suffix(".inference.txt")
            lines = [
                f"XML: {args.xml}",
                f"Image: {args.image}",
                "",
                f"CER (OCR): {results['cer_ocr']:.4f}",
                f"CER (corrected): {results['cer_corrected']:.4f}",
                "",
                "--- Ground Truth ---",
                results["ground_truth"],
                "",
                "--- OCR Prediction ---",
                results["full_text_ocr"],
                "",
                "--- Corrected ---",
                results["full_text_corrected"],
            ]
            out_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Results saved to {out_path}")

    else:
        if args.data == "raw":
            train_loader, val_loader, test_loader = build_dataloaders(
                args.raw_dir, exclude=args.exclude, batch_size=16, num_workers=0,
            )
        elif args.data == "synthetic":
            train_loader, val_loader, test_loader = build_dataloaders(
                synthetic_dir=args.synthetic_dir, batch_size=16, num_workers=0,
            )
        else:
            train_loader, val_loader, test_loader = build_dataloaders(
                args.raw_dir, exclude=args.exclude, synthetic_dir=args.synthetic_dir,
                batch_size=16, num_workers=0,
            )
        split_dataset = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split].dataset
        print(f"Running inference on {args.split} split ({len(split_dataset)} samples)...")

        results = inferencer.full_inference_from_dataset(split_dataset)
        print(f"\nCER (OCR): {results['cer_ocr']:.4f}")
        print(f"CER (corrected): {results['cer_corrected']:.4f}")
        print(f"\nSample predictions:")
        for sample in results["samples"][:5]:
            print(f"truth: {sample['truth']}")
            print(f"ocr: {sample['ocr']}")
            print(f"corrected: {sample['corrected']}")
            print()

        if args.save_text:
            out_path = Path(f"inference_{args.data}_{args.split}.txt")
            lines = [
                f"Data: {args.data}",
                f"Split: {args.split}",
                f"Samples: {len(results['samples'])}",
                "",
                f"CER (OCR): {results['cer_ocr']:.4f}",
                f"CER (corrected): {results['cer_corrected']:.4f}",
                "",
            ]
            for i, sample in enumerate(results["samples"], 1):
                lines += [
                    f"--- Sample {i} ---",
                    f"Truth: {sample['truth']}",
                    f"OCR: {sample['ocr']}",
                    f"Corrected: {sample['corrected']}",
                    "",
                ]
            out_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Results saved to {out_path}")

