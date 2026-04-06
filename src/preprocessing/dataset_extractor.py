import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.preprocessing.constants import (
    ACTIVITY_MAP,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    TEST_SUBJECTS,
    ACTIVITY_TO_IDX,
)
from src.eda.exploratory_analysis import run_eda

# From the dataset readme:

# The IMU sensory data contains the following columns:
# –1 temperature (°C)
# –2-4 3D-acceleration data (ms-2),  scale: 16g, resolution: 13-bit
# –5-7 3D-acceleration data (ms-2),  scale: 6g, resolution: 13-bit
# –8-10 3D-gyroscope data (rad/s)
# –11-13 3D-magnetometer data (μT)
# –14-17 orientation (invalid in this data collection)

# The dataset readme mentions that the 6g accelerometer is not precisely
# calibrated with the first one (16g), and the first accelerometer data is
# recommended to use
# For this reason, the 6g accelerometer data is dropped from the dataset
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
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
]


def get_column_names():
    cols = ["timestamp", "activity_id", "heart_rate"]
    for pos in ["hand", "chest", "ankle"]:
        for comp in IMU_COMPONENTS:
            cols.append(f"{pos}_{comp}")
    return cols


def extract_subject_data(
    file_path: str, subject_id: int, drop_features: list[str] = None
) -> pd.DataFrame:
    """
    Cleaning steps from the dataset README:
      1. Discard activity_id == 0 (transient / unlabeled periods)
      2. Drop orientation columns (flagged as invalid in this data collection)
      3. Drop 6g accelerometer (can saturate during high-impact moves; not precisely calibrated relative to ±16g)
      4. Manual Feature Selection: Drop any features specified by the user.
      5. Interpolate heart rate (HR monitor runs at ~9 Hz vs IMU at 100 Hz, so most rows have NaN heart_rate)
      6. Forward/back-fill IMU (occasional wireless packet drops)
    """
    print(f"  Loading {file_path} ...")
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df.columns = get_column_names()
    df["subject_id"] = subject_id

    # Discard transient activity
    df = df[df["activity_id"] != 0].copy()

    # Drop orientation columns
    orient_cols = [c for c in df.columns if "orient" in c]
    df = df.drop(columns=orient_cols)

    # Drop 6g accelerometer channels
    acc6_cols = [c for c in df.columns if "acc6" in c]
    df = df.drop(columns=acc6_cols)

    # Drop manually specified features
    if drop_features:
        cols_to_drop = [
            c for c in df.columns if any(feat in c for feat in drop_features)
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    # Heart rate: linear interpolation then forward/back-fill for edge NaNs
    if "heart_rate" in df.columns:
        df["heart_rate"] = df["heart_rate"].interpolate().ffill().bfill()

    # IMU sensors: forward/back-fill for wireless packet drops
    sensor_cols = [
        c for c in df.columns if c not in ("timestamp", "activity_id", "subject_id")
    ]
    df[sensor_cols] = df[sensor_cols].ffill().bfill()
    return df


def get_cleaned_data(drop_features: list[str] = None) -> pd.DataFrame:
    root_dir = Path("PAMAP2_Dataset")
    all_frames = []
    subjects = [101, 102, 103, 104, 105, 106, 107, 108, 109]
    for sid in subjects:
        for folder in ["Protocol", "Optional"]:
            p = root_dir / folder / f"subject{sid}.dat"
            if p.exists():
                all_frames.append(extract_subject_data(p, sid, drop_features))
    return pd.concat(all_frames, ignore_index=True)


def save_tabular_splits(df: pd.DataFrame, output_subdir: str):
    root_dir = Path("PAMAP2_Dataset")
    output_base = root_dir / "processed" / output_subdir
    output_base.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        c for c in df.columns if c not in ("timestamp", "activity_id", "subject_id")
    ]
    scaler = StandardScaler()
    train_mask = df["subject_id"].isin(TRAIN_SUBJECTS)
    scaler.fit(df.loc[train_mask, feature_cols])
    df[feature_cols] = scaler.transform(df[feature_cols])

    splits = {"train": TRAIN_SUBJECTS, "val": VAL_SUBJECTS, "test": TEST_SUBJECTS}
    for name, sids in splits.items():
        split_df = df[df["subject_id"].isin(sids)]
        X = split_df[feature_cols].values.astype(np.float32)
        y = np.array(
            [ACTIVITY_TO_IDX[a] for a in split_df["activity_id"]], dtype=np.int32
        )
        np.save(output_base / f"{name}_X.npy", X)
        np.save(output_base / f"{name}_y.npy", y)
        print(f" Saved tabular {output_subdir}/{name}: X={X.shape}, y={y.shape}")


def run_pipeline_step(drop_features: list[str] = None, output_subdir: str = "normal"):
    print(f"--- Processing {output_subdir} ---")
    df = get_cleaned_data(drop_features)

    eda_dir = Path("PAMAP2_Dataset") / (
        "feature_selection_eda_plots"
        if output_subdir == "feature_selection"
        else "eda_plots"
    )
    run_eda(df, eda_dir)

    save_tabular_splits(df, output_subdir)
    return df


def main():
    run_pipeline_step(None, "normal")
    run_pipeline_step(MANUAL_FEATURES_TO_DROP, "feature_selection")


if __name__ == "__main__":
    main()
