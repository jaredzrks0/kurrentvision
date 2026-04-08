import argparse
from pathlib import Path

from kraken import binarization, pageseg
from PIL import Image, ImageDraw


def _merge_overlapping(bboxes, overlap_thresh=0.5):
    merged = []
    used = [False] * len(bboxes)
    for i, (ax1, ay1, ax2, ay2) in enumerate(bboxes):
        if used[i]:
            continue
        group = [bboxes[i]]
        used[i] = True
        for j, (bx1, by1, bx2, by2) in enumerate(bboxes):
            if used[j]:
                continue
            # Check vertical overlap ratio
            overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
            height_a = ay2 - ay1
            height_b = by2 - by1
            ratio = overlap_y / min(height_a, height_b) if min(height_a, height_b) > 0 else 0
            if ratio >= overlap_thresh:
                group.append(bboxes[j])
                used[j] = True
        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[2] for b in group)
        y2 = max(b[3] for b in group)
        merged.append((x1, y1, x2, y2))
    return sorted(merged, key=lambda b: b[1])


def detect_lines(image_path: str | Path):
    image = Image.open(image_path)
    bw = binarization.nlbin(image)
    seg = pageseg.segment(bw)

    bboxes = []
    for line in seg.lines:
        x1, y1, x2, y2 = line.bbox
        bboxes.append((x1, y1, x2, y2))

    bboxes = _merge_overlapping(bboxes)
    return image.convert("RGB"), bboxes


def draw_boxes(image: Image.Image, bboxes) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), str(i), fill="red")
    return annotated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect text line bounding boxes in an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", default=None, help="Path to save annotated image (default: <image>_boxes.png)")
    args = parser.parse_args()

    image, bboxes = detect_lines(args.image)
    annotated = draw_boxes(image, bboxes)

    out_path = Path(args.output) if args.output else Path(args.image).with_name(Path(args.image).stem + "_boxes.png")
    annotated.save(out_path)
    print(f"Detected {len(bboxes)} lines. Saved to {out_path}")
