import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing.constants import IDX_TO_ACTIVITY
from src.models.utils.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_pr_curves,
    save_metrics_report,
)


def run_rf(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = [IDX_TO_ACTIVITY[i] for i in range(len(IDX_TO_ACTIVITY))]

    print(f" Loading data from {data_dir} ...")
    X_train = np.load(data_dir / "train_X.npy")
    y_train = np.load(data_dir / "train_y.npy")
    X_val = np.load(data_dir / "val_X.npy")
    y_val = np.load(data_dir / "val_y.npy")
    X_test = np.load(data_dir / "test_X.npy")
    y_test = np.load(data_dir / "test_y.npy")

    # Model reference: https://github.com/andreasKyratzis/PAMAP2-Physical-Activity-Monitoring-Data-Analysis-and-ML/blob/master/pamap2.ipynb
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")

    print(" Training Random Forest ...")
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    clf.fit(X_train_val, y_train_val)

    all_metrics = []
    for name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)

        m = calculate_metrics(y, y_pred, y_prob, name)
        all_metrics.append(m)

        plot_confusion_matrix(y, y_pred, name, output_dir, class_names)
        plot_pr_curves(y, y_prob, name, output_dir, class_names)
        print(f"  {name.upper()}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}")

    save_metrics_report(all_metrics, output_dir, "Random Forest")


def main():
    processed_base = Path("PAMAP2_Dataset/processed")
    results_base = Path("results/rf")
    for subdir in ["normal", "feature_selection"]:
        run_rf(processed_base / subdir, results_base / subdir)


if __name__ == "__main__":
    main()
