import matplotlib.pyplot as plt

from pathlib import Path

def _edit_distance(prediction: str, target: str) -> int:
    costs = list(range(len(target) + 1))
    for row, source_char in enumerate(prediction, 1):
        diagonal, costs[0] = costs[0], row
        for col, target_char in enumerate(target, 1):
            diagonal, costs[col] = costs[col], diagonal if source_char == target_char else 1 + min(diagonal, costs[col], costs[col - 1])
    return costs[-1]


def cer(preds: list[str], targets: list[str]) -> float:
    total_edits = sum(_edit_distance(pred, true) for pred, true in zip(preds, targets))
    total_chars = sum(len(target) for target in targets)
    return total_edits / total_chars if total_chars > 0 else 0.0

def save_training_plots(history: dict, save_dir: Path) -> None:
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