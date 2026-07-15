"""
main.py
=======
Runs the full experiment required by the assignment:

  1. Loads CIFAR-10 with an identical data pipeline for both models.
  2. Trains Model 1 (PlainCNN) and Model 2 (ResNetCNN) with IDENTICAL
     hyperparameters (optimizer, LR schedule, batch size, epochs, augmentation).
  3. Produces every requested deliverable into ./outputs/:
       - loss curves, accuracy curves (PNG)
       - confusion matrices for both models (PNG)
       - comparison table (CSV + Markdown)
       - training histories (JSON)
       - final trained weights (.pt)
       - a run_log.txt with everything printed to console

Usage:
    python main.py                       # full run (default settings)
    python main.py --epochs 5 --n 2      # quick smoke-test run
    python main.py --epochs 60 --n 3     # closer to paper-quality ResNet-20

See README.md for details and expected runtime.
"""

import argparse
import os
import time
import torch

from src.models import PlainCNN, ResNetCNN
from src.data import get_dataloaders, CLASS_NAMES
from src.train import train_model, evaluate_test
from src.utils import (count_parameters, plot_curves, plot_confusion_matrix,
                        save_comparison_table, save_history_json)


def parse_args():
    p = argparse.ArgumentParser(description="Plain CNN vs Residual CNN on CIFAR-10")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./outputs")
    p.add_argument("--n", type=int, default=3,
                    help="Blocks per stage. n=3 -> 20-layer networks (ResNet-20 style).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true",
                    help="Disable random crop / flip augmentation.")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "run_log.txt")
    log_file = open(log_path, "a")

    def log(msg):
        print(msg)
        log_file.write(str(msg) + "\n")
        log_file.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    if device.type == "cpu":
        log("WARNING: no GPU detected. Training will be slow on CPU. "
            "Consider running this on Google Colab (free GPU) for the real run.")

    set_seed(args.seed)

    # ---- Identical data pipeline for both models ----
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        val_fraction=args.val_fraction, augment=not args.no_augment,
        seed=args.seed,
    )
    log(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} "
        f"| Test batches: {len(test_loader)}")

    lr_milestones = (int(args.epochs * 0.5), int(args.epochs * 0.75))

    results_rows = []
    histories = {}

    for model_name, model_cls in [("PlainCNN", PlainCNN), ("ResNetCNN", ResNetCNN)]:
        log(f"\n===== Training {model_name} (n={args.n}) =====")
        set_seed(args.seed)  # re-seed so both models start from comparable init noise
        model = model_cls(n=args.n)
        n_params = count_parameters(model)
        log(f"{model_name} trainable parameters: {n_params:,}")

        history, train_time = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, lr_milestones=lr_milestones,
            log_fn=log,
        )
        histories[model_name] = history

        test_acc, y_true, y_pred = evaluate_test(model, test_loader, device)
        log(f"{model_name} FINAL TEST ACCURACY: {test_acc:.4f}")
        log(f"{model_name} training time: {train_time:.1f}s "
            f"({train_time/60:.1f} min)")

        # Save weights
        torch.save(model.state_dict(),
                   os.path.join(args.out_dir, f"{model_name}_weights.pt"))

        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred, CLASS_NAMES,
            title=f"{model_name} Confusion Matrix (Test Set)",
            out_path=os.path.join(args.out_dir, f"{model_name}_confusion_matrix.png"),
        )

        save_history_json(history, os.path.join(args.out_dir, f"{model_name}_history.json"))

        results_rows.append({
            "Model": model_name,
            "Trainable Parameters": f"{n_params:,}",
            "Final Train Acc": f"{history['train_acc'][-1]:.4f}",
            "Final Val Acc": f"{history['val_acc'][-1]:.4f}",
            "Test Accuracy": f"{test_acc:.4f}",
            "Training Time (s)": f"{train_time:.1f}",
        })

    # ---- Combined curves (both models on the same plot) ----
    plot_curves(histories["PlainCNN"], histories["ResNetCNN"],
                "PlainCNN", "ResNetCNN",
                out_path_prefix=os.path.join(args.out_dir, "comparison"))

    # ---- Comparison table ----
    md_table = save_comparison_table(
        results_rows,
        out_csv_path=os.path.join(args.out_dir, "comparison_table.csv"),
        out_md_path=os.path.join(args.out_dir, "comparison_table.md"),
    )
    log("\n===== COMPARISON TABLE =====")
    log(md_table)

    log(f"\nAll outputs saved to: {os.path.abspath(args.out_dir)}")
    log_file.close()


if __name__ == "__main__":
    main()
