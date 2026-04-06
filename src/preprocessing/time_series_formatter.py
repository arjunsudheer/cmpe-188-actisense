import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing.constants import (
    ACTIVITY_MAP,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    TEST_SUBJECTS,
    ACTIVITY_TO_IDX,
)


def create_sliding_windows(
    df: pd.DataFrame, window_size: int = 128, step_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = [
        c for c in df.columns if c not in ("timestamp", "activity_id", "subject_id")
    ]
    X_list, y_list = [], []
    for (subj, act), group in df.groupby(["subject_id", "activity_id"]):
        data = group[feature_cols].values
        if len(data) < window_size:
            continue
        for start in range(0, len(data) - window_size + 1, step_size):
            X_list.append(data[start : start + window_size])
            y_list.append(ACTIVITY_TO_IDX[act])
    if not X_list:
        return np.array([]), np.array([])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def run_windowing(df: pd.DataFrame, output_subdir: str):
    output_base = Path("PAMAP2_Dataset/processed") / output_subdir
    output_base.mkdir(parents=True, exist_ok=True)

    splits = {"train": TRAIN_SUBJECTS, "val": VAL_SUBJECTS, "test": TEST_SUBJECTS}
    for name, sids in splits.items():
        split_df = df[df["subject_id"].isin(sids)]
        X, y = create_sliding_windows(split_df)
        if X.size > 0:
            np.save(output_base / f"{name}_windows_X.npy", X)
            np.save(output_base / f"{name}_windows_y.npy", y)
            print(f" Saved windows {output_subdir}/{name}: X={X.shape}, y={y.shape}")


def main():
    # This main is just for standalone testing, normally called from main.py
    processed_dir = Path("PAMAP2_Dataset/processed")
    for subdir in ["normal", "feature_selection"]:
        # Mocking or loading if it existed, but better to call from main.py
        pass


if __name__ == "__main__":
    main()
