import time
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing.constants import IDX_TO_ACTIVITY, ACTIVITY_MAP
from src.models.utils.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_pr_curves,
    save_metrics_report,
)


def _build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=20,  # Used to prevent overfitting
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",  # Used to handle class imbalance
        random_state=42,
        n_jobs=-1,
    )


def _stream_predict(
    clf: RandomForestClassifier, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(X_test)
    n_classes = len(ACTIVITY_MAP)
    seen_classes = clf.classes_

    # Batch prediction for efficiency
    print(f"Calculating results for {n} samples")
    y_pred = clf.predict(X_test)
    y_prob_seen = clf.predict_proba(X_test)
    y_prob = np.zeros((n, n_classes), dtype=np.float64)
    y_prob[:, seen_classes] = y_prob_seen

    # Timing subset to estimate streaming latency
    n_time = min(n, 1000)
    print(f"Measuring streaming latency on {n_time} samples")
    latencies_ms = []
    for i in range(n_time):
        sample = X_test[i : i + 1]
        t0 = time.perf_counter()
        _ = clf.predict(sample)
        _ = clf.predict_proba(sample)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    avg_ms = sum(latencies_ms) / n_time
    total_ms = avg_ms * n
    return y_pred, y_prob, {"avg_ms": avg_ms, "total_ms": total_ms, "n_samples": n}


def run_rf(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = [ACTIVITY_MAP[IDX_TO_ACTIVITY[i]] for i in range(len(ACTIVITY_MAP))]

    print(f"Loading ML features from {data_dir} ")
    X_train = np.load(data_dir / "train_ml_X.npy")
    y_train = np.load(data_dir / "train_y.npy")
    X_test = np.load(data_dir / "test_ml_X.npy")
    y_test = np.load(data_dir / "test_y.npy")

    clf = _build_rf()

    print("Training Random Forest")
    clf.fit(X_train, y_train)

    all_metrics = []
    for name, X, y in [
        ("train", X_train, y_train),
        ("test", X_test, y_test),
    ]:
        # Evaluation
        y_pred = clf.predict(X)
        y_prob_seen = clf.predict_proba(X)
        y_prob = np.zeros((len(X), len(ACTIVITY_MAP)), dtype=np.float64)
        y_prob[:, clf.classes_] = y_prob_seen

        m = calculate_metrics(y, y_pred, y_prob, name)
        all_metrics.append(m)
        plot_confusion_matrix(y, y_pred, name, output_dir, class_names)
        plot_pr_curves(y, y_prob, name, output_dir, class_names)
        print(f"{name.upper()}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}")

    # Streaming prediction on test set
    print("Streaming test predictions")
    _, _, timing = _stream_predict(clf, X_test)
    print(
        f"Streaming: avg={timing['avg_ms']:.4f}ms/window, "
        f"total={timing['total_ms']:.2f}ms over {timing['n_samples']} windows"
    )
    save_metrics_report(all_metrics, output_dir, "Random Forest", timing=timing)


def main():
    processed_base = Path("PAM2_Dataset/processed")
    results_base = Path("results/rf")
    for subdir in ["normal", "feature_selection"]:
        run_rf(processed_base / subdir, results_base / subdir)


if __name__ == "__main__":
    main()
