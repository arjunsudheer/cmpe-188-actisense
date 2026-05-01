import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Precision loss occurred in moment calculation",
)
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.preprocessing.constants import (
    ALL_SUBJECTS,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    TEST_SUBJECTS,
    ACTIVITY_MAP,
    ACTIVITY_TO_IDX,
)
from src.eda.exploratory_analysis import run_eda
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy as sp_entropy
from sklearn.feature_selection import VarianceThreshold
import joblib

IMU_COMPONENTS = [
    "temp",
    "acc16_x",
    "acc16_y",
    "acc16_z",
    "acc6_x",
    "acc6_y",
    "acc6_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "orient_w",
    "orient_x",
    "orient_y",
    "orient_z",
]

MANUAL_FEATURES_TO_DROP = [
    "chest_mag_x",  # High correlation with chest_mag_y
    "chest_mag_z",  # High correlation with chest_acc16_z
    "ankle_mag_y",  # Overlaps with chest_mag_z (negative correlation)
]


def get_column_names():
    imu_cols = [
        "temp",
        "acc16_x",
        "acc16_y",
        "acc16_z",
        "acc6_x",
        "acc6_y",
        "acc6_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "mag_x",
        "mag_y",
        "mag_z",
        "orient1",
        "orient2",
        "orient3",
        "orient4",
    ]
    col_names = ["timestamp", "activity_id", "heart_rate"]
    for s in ["hand", "chest", "ankle"]:
        col_names += [f"{s}_{c}" for c in imu_cols]
    return col_names


def get_cleaned_data():
    root_dir = Path("PAMAP2_Dataset")
    col_names = get_column_names()
    frames = []

    for sid in ALL_SUBJECTS:
        files = list(root_dir.glob(f"*/subject{sid}.dat"))
        if not files:
            continue
        dfs = [pd.read_csv(f, sep=r"\s+", header=None, names=col_names) for f in files]
        sub_df = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
        sub_df["subject_id"] = sid
        frames.append(sub_df)

    raw = pd.concat(frames, ignore_index=True).drop(columns=["timestamp"])

    # Keep only 12 protocol activities
    raw = raw[raw["activity_id"].isin(ACTIVITY_MAP.keys())].copy()
    raw["label"] = raw["activity_id"].map(ACTIVITY_TO_IDX)
    raw.drop(columns=["activity_id"], inplace=True)

    # Discard non-protocol IMU channels
    cols_to_keep = [
        c
        for c in raw.columns
        if "orient" not in c and "acc6" not in c and "temp" not in c
    ]
    df = raw[cols_to_keep].copy()
    return df


def impute(df_split, medians=None):
    df_split = df_split.copy()
    sensor_cols = [
        c for c in df_split.columns if c not in ["label", "subject_id", "heart_rate"]
    ]

    for sid in df_split["subject_id"].unique():
        mask = df_split["subject_id"] == sid
        # Heart rate: Forward-fill then backward-fill within each subject
        df_split.loc[mask, "heart_rate"] = (
            df_split.loc[mask, "heart_rate"].ffill().bfill()
        )
        # IMU: Linear interpolation with max gap of 5 rows
        df_split.loc[mask, sensor_cols] = df_split.loc[mask, sensor_cols].interpolate(
            method="linear", limit=5
        )

    if medians is None:
        medians = df_split[sensor_cols + ["heart_rate"]].median()

    # Residual NaN fill with medians
    for col in sensor_cols + ["heart_rate"]:
        df_split[col] = df_split[col].fillna(medians[col])

    return df_split, medians


def create_windows(df_split, window_size=128, step_size=64, min_purity=0.9):
    feature_cols = [c for c in df_split.columns if c not in ["label", "subject_id"]]
    all_windows, all_labels = [], []

    for sid in df_split["subject_id"].unique():
        sub = df_split[df_split["subject_id"] == sid]
        data = sub[feature_cols].values
        labels = sub["label"].values

        for start in range(0, len(data) - window_size + 1, step_size):
            chunk = data[start : start + window_size]
            lbls = labels[start : start + window_size]
            vals, counts = np.unique(lbls, return_counts=True)
            majority = vals[np.argmax(counts)]
            if counts.max() / window_size >= min_purity:
                all_windows.append(chunk)
                all_labels.append(majority)

    return np.array(all_windows), np.array(all_labels)


# Hz rate
FS = 100


def extract_features_vectorized(X, acc16_idx):
    N, T, C = X.shape

    # Time domain (N, C, 10)
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    rms = np.sqrt(np.mean(X**2, axis=1))
    sk = skew(X, axis=1)
    kt = kurtosis(X, axis=1)
    mad = np.mean(np.abs(X - mu), axis=1)
    zcr = np.sum(np.diff(np.sign(X - mu), axis=1) != 0, axis=1) / T

    time_feats = np.stack(
        [mu.squeeze(1), std, mn, mx, mx - mn, rms, sk, kt, mad, zcr], axis=2
    )

    # Frequency domain (N, C, 8)
    X_f = rfft(X - mu, axis=1)
    pwr = np.abs(X_f) ** 2
    freqs = rfftfreq(T, d=1 / FS)

    total_pwr = pwr.sum(axis=1)
    norm_pwr = pwr / (total_pwr[:, np.newaxis, :] + 1e-8)
    # Entropy calculation
    ent = -np.sum(norm_pwr * np.log(norm_pwr + 1e-8), axis=1)

    dom_idx = np.argmax(pwr, axis=1)
    dom_freq = freqs[dom_idx]
    dom_pwr = np.take_along_axis(pwr, dom_idx[:, np.newaxis, :], axis=1).squeeze(1)

    # Sub-bands
    b1 = pwr[:, (freqs > 0) & (freqs <= 1), :].sum(axis=1)
    b2 = pwr[:, (freqs > 1) & (freqs <= 3), :].sum(axis=1)
    b3 = pwr[:, (freqs > 3) & (freqs <= 10), :].sum(axis=1)
    b4 = pwr[:, (freqs > 10) & (freqs <= 25), :].sum(axis=1)

    freq_feats = np.stack([dom_freq, dom_pwr, ent, total_pwr, b1, b2, b3, b4], axis=2)

    all_channel_feats = np.concatenate([time_feats, freq_feats], axis=2).reshape(N, -1)

    # Cross-axis correlations (N, 9)
    corr_feats = []
    for i, j, k in acc16_idx:
        for a, b in [(i, j), (i, k), (j, k)]:
            xa, xb = X[:, :, a], X[:, :, b]
            ma, mb = xa.mean(axis=1, keepdims=True), xb.mean(axis=1, keepdims=True)
            sa, sb = xa.std(axis=1), xb.std(axis=1)
            c = np.mean((xa - ma) * (xb - mb), axis=1) / (sa * sb + 1e-8)
            corr_feats.append(c)

    corr_feats = np.stack(corr_feats, axis=1)

    return np.nan_to_num(np.concatenate([all_channel_feats, corr_feats], axis=1))


def scale_seq(X, sc):
    n, t, c = X.shape
    return sc.transform(X.reshape(-1, c)).reshape(n, t, c)


def run_pipeline_step(drop_features: list = None, output_subdir: str = "normal"):
    print(f"Processing {output_subdir}")
    df = get_cleaned_data()

    if drop_features:
        cols_to_drop = [c for c in df.columns if any(f in c for f in drop_features)]
        if cols_to_drop:
            print(f"Dropping features: {cols_to_drop}")
            df.drop(columns=cols_to_drop, inplace=True)

    # EDA needs clean data
    df_eda = df.copy()
    sensor_cols = [c for c in df_eda.columns if c not in ("label", "subject_id")]
    for sid in df_eda["subject_id"].unique():
        mask = df_eda["subject_id"] == sid
        df_eda.loc[mask, sensor_cols] = (
            df_eda.loc[mask, sensor_cols].ffill().bfill().interpolate()
        )
    df_eda.fillna(df_eda.median(), inplace=True)

    eda_dir = Path("PAMAP2_Dataset") / (
        "feature_selection_eda_plots"
        if output_subdir == "feature_selection"
        else "eda_plots"
    )
    idx_to_act = {v: k for k, v in ACTIVITY_TO_IDX.items()}
    df_eda["activity_id"] = df_eda["label"].map(idx_to_act)
    run_eda(df_eda, eda_dir)

    # Split data
    df_train = df[df["subject_id"].isin(TRAIN_SUBJECTS)].reset_index(drop=True)
    df_val = df[df["subject_id"].isin(VAL_SUBJECTS)].reset_index(drop=True)
    df_test = df[df["subject_id"].isin(TEST_SUBJECTS)].reset_index(drop=True)

    # Impute missing values
    df_train, train_medians = impute(df_train)
    df_val, _ = impute(df_val, medians=train_medians)
    df_test, _ = impute(df_test, medians=train_medians)

    # Windowing
    X_train, y_train = create_windows(df_train)
    X_val, y_val = create_windows(df_val)
    X_test, y_test = create_windows(df_test)

    # ML Pipeline
    feature_cols = [c for c in df.columns if c not in ["label", "subject_id"]]
    acc16_idx = []
    for s in ["hand", "chest", "ankle"]:
        try:
            acc16_idx.append(
                (
                    feature_cols.index(f"{s}_acc16_x"),
                    feature_cols.index(f"{s}_acc16_y"),
                    feature_cols.index(f"{s}_acc16_z"),
                )
            )
        except ValueError:
            pass

    print(f"Extracting ML features (vectorized) for {len(X_train)} windows")
    X_train_ml = extract_features_vectorized(X_train, acc16_idx)
    X_val_ml = extract_features_vectorized(X_val, acc16_idx)
    X_test_ml = extract_features_vectorized(X_test, acc16_idx)

    # Scaling
    scaler_ml = StandardScaler()
    X_train_ml = scaler_ml.fit_transform(X_train_ml)
    X_val_ml = scaler_ml.transform(X_val_ml)
    X_test_ml = scaler_ml.transform(X_test_ml)

    # DL Pipeline
    print("Scaling DL sequences")
    scaler_dl = StandardScaler()
    scaler_dl.fit(X_train.reshape(-1, X_train.shape[2]))

    X_train_dl = scale_seq(X_train, scaler_dl)
    X_val_dl = scale_seq(X_val, scaler_dl)
    X_test_dl = scale_seq(X_test, scaler_dl)

    # Save everything
    output_base = Path("PAMAP2_Dataset/processed") / output_subdir
    output_base.mkdir(parents=True, exist_ok=True)

    np.save(output_base / "train_ml_X.npy", X_train_ml)
    np.save(output_base / "train_y.npy", y_train)
    np.save(output_base / "val_ml_X.npy", X_val_ml)
    np.save(output_base / "val_y.npy", y_val)
    np.save(output_base / "test_ml_X.npy", X_test_ml)
    np.save(output_base / "test_y.npy", y_test)

    np.save(output_base / "train_windows_X.npy", X_train_dl)
    np.save(output_base / "val_windows_X.npy", X_val_dl)
    np.save(output_base / "test_windows_X.npy", X_test_dl)

    joblib.dump(scaler_ml, output_base / "scaler_ml.pkl")
    joblib.dump(scaler_dl, output_base / "scaler_dl.pkl")

    print(f"Done {output_subdir}: ML_X={X_train_ml.shape}, DL_X={X_train_dl.shape}")
    return df


if __name__ == "__main__":
    # Run the pipeline for both 'normal' and 'feature_selection' (manual drops)
    # The 'normal' run now has temperature channels dropped by default in get_cleaned_data
    run_pipeline_step(output_subdir="normal")
    run_pipeline_step(
        drop_features=MANUAL_FEATURES_TO_DROP, output_subdir="feature_selection"
    )
