import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, split_name: str
) -> dict:
    return {
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "log_loss": log_loss(y_true, y_prob, labels=range(y_prob.shape[1])),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    output_dir: Path,
    class_names: list,
):
    cm = confusion_matrix(y_true, y_pred)
    # Ensure display_labels matches the unique classes present in y_true/y_pred
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    display_labels = [class_names[i] for i in unique_labels]

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {split_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"cm_{split_name.lower()}.png", dpi=150)
    plt.close()


def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    split_name: str,
    output_dir: Path,
    class_names: list,
):
    n_classes = y_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(15, 10))
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curves - {split_name}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"pr_{split_name.lower()}.png", dpi=150)
    plt.close()


def save_metrics_report(metrics_list: list[dict], output_dir: Path, model_name: str):
    report_path = output_dir / "metrics_report.txt"
    with open(report_path, "w") as f:
        for m in metrics_list:
            f.write(f"[{m['split'].upper()}]\n")
            f.write(f"  Accuracy  : {m['accuracy']:.4f}\n")
            f.write(f"  Precision : {m['precision']:.4f}\n")
            f.write(f"  Recall    : {m['recall']:.4f}\n")
            f.write(f"  F1-Score  : {m['f1']:.4f}\n")
            f.write(f"  Log-Loss  : {m['log_loss']:.4f}\n\n")
    print(f" Saved report to {report_path}")
