"""
utils.py
========
Helper functions for the deliverables the assignment asks for:
- trainable parameter counts
- loss / accuracy curve plots
- confusion matrix plot
- comparison table (saved as CSV + printed as markdown)
"""

import json
import numpy as np
import matplotlib.pyplot as plt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_curves(history_a, history_b, label_a, label_b, out_path_prefix):
    """Saves two figures: loss curves and accuracy curves, each comparing
    both models' train and validation lines."""

    # --- Loss curves ---
    plt.figure(figsize=(7, 5))
    plt.plot(history_a["train_loss"], label=f"{label_a} - train", linestyle="-")
    plt.plot(history_a["val_loss"], label=f"{label_a} - val", linestyle="--")
    plt.plot(history_b["train_loss"], label=f"{label_b} - train", linestyle="-")
    plt.plot(history_b["val_loss"], label=f"{label_b} - val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_loss_curves.png", dpi=150)
    plt.close()

    # --- Accuracy curves ---
    plt.figure(figsize=(7, 5))
    plt.plot(history_a["train_acc"], label=f"{label_a} - train", linestyle="-")
    plt.plot(history_a["val_acc"], label=f"{label_a} - val", linestyle="--")
    plt.plot(history_b["train_acc"], label=f"{label_b} - train", linestyle="-")
    plt.plot(history_b["val_acc"], label=f"{label_b} - val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_accuracy_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, title, out_path):
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black",
                      fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return cm


def save_comparison_table(rows, out_csv_path, out_md_path):
    """
    rows: list of dicts, one per model, e.g.
        {"Model": "Plain CNN", "Test Accuracy": 0.87, ...}
    """
    import csv
    keys = list(rows[0].keys())

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["| " + " | ".join(keys) + " |",
                "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        md_lines.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    md_table = "\n".join(md_lines)

    with open(out_md_path, "w") as f:
        f.write(md_table)

    return md_table


def save_history_json(history, out_path):
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
