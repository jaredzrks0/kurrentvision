import argparse

from modeling.kraken_finetune import run, DATASETS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Kraken on Kurrent handwriting data")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()) + ["all"], default=["all"])
    parser.add_argument("--output-dir", default="models/kraken")
    parser.add_argument("--seg-model", default=None, help="Base segmentation model (.mlmodel or .ckpt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--augment", action="store_true", default=True)
    args = parser.parse_args()

    run(
        datasets=args.datasets,
        output_dir=args.output_dir,
        seg_model=args.seg_model,
        epochs=args.epochs,
        augment=args.augment,
    )
