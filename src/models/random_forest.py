import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


def _get_feature_names(n_features):
    # heart_rate (1) + 3 IMUs * 9 components = 28 base channels
    base_components = ["acc16_x", "acc16_y", "acc16_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]
    base_channels = ["heart_rate"]
    for s in ["hand", "chest", "ankle"]:
        for c in base_components:
            base_channels.append(f"{s}_{c}")
            
    time_names = ["mean", "std", "min", "max", "range", "rms", "skew", "kurtosis", "mad", "zcr"]
    freq_names = ["dom_freq", "dom_pwr", "entropy", "total_pwr", "b1", "b2", "b3", "b4"]
    n_per_chan = len(time_names) + len(freq_names) # 18
    n_corr = 9 # 3 IMUs * 3 pairs
    
    # Try different combinations to match n_features
    # 1. Everything (Normal)
    # 2. Dropping gyro (Legacy selection)
    # 3. Dropping overlapping sensors (New selection: chest_mag_x, chest_mag_z, ankle_mag_y)
    
    current_drop = ["chest_mag_x", "chest_mag_z", "ankle_mag_y"]
    
    configs = [
        ("all", base_channels),
        ("no_gyro", [c for c in base_channels if "gyro" not in c]),
        ("overlapping_drop", [c for c in base_channels if c not in current_drop]),
    ]
    
    channels = base_channels
    for name, chan_list in configs:
        if len(chan_list) * n_per_chan + n_corr == n_features:
            channels = chan_list
            break
            
    feat_names = []
    for col in channels:
        for tn in time_names:
            feat_names.append(f"{col}_{tn}")
        for fn in freq_names:
            feat_names.append(f"{col}_{fn}")
            
    for s in ["hand", "chest", "ankle"]:
        feat_names.append(f"{s}_acc16_xy_corr")
        feat_names.append(f"{s}_acc16_xz_corr")
        feat_names.append(f"{s}_acc16_yz_corr")
        
    return feat_names


def _plot_feature_importance(clf, output_dir, top_n=30):
    importance = clf.feature_importances_
    try:
        feat_names = _get_feature_names(len(importance))
        if len(feat_names) != len(importance):
             print(f"Warning: Feature names count ({len(feat_names)}) does not match model features ({len(importance)})")
             feat_names = [f"f{i}" for i in range(len(importance))]
    except Exception as e:
        print(f"Error generating feature names: {e}")
        feat_names = [f"f{i}" for i in range(len(importance))]

    df_imp = pd.DataFrame({"feature": feat_names, "importance": importance})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(12, 10))
    sns.barplot(data=df_imp, x="importance", y="feature", palette="magma", hue="feature", legend=False)
    plt.title(f"Top {top_n} Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()

    # Also save as CSV for bookkeeping
    df_imp.to_csv(output_dir / "feature_importance.csv", index=False)


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

    # Feature Importance
    _plot_feature_importance(clf, output_dir)

    save_metrics_report(all_metrics, output_dir, "Random Forest", timing=timing)


def main():
    processed_base = Path("PAMAP2_Dataset/processed")
    results_base = Path("results/rf")
    for subdir in ["normal", "feature_selection"]:
        run_rf(processed_base / subdir, results_base / subdir)


if __name__ == "__main__":
    main()
