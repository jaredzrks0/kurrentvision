import argparse

from src.modeling.kraken_predict import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict text line bounding boxes with a Kraken segmentation model")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("model", help="Path to Kraken model (.ckpt or .mlmodel)")
    parser.add_argument("--output", default=None, help="Path to save annotated image")
    args = parser.parse_args()

    run(args.image, args.model, args.output)
