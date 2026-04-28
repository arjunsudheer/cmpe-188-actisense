# 12 protocol activities performed across sessions
ACTIVITY_MAP = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic walking",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    24: "rope jumping",
}

# Label mapping for models (0-11)
ACTIVITY_TO_IDX = {act: i for i, act in enumerate(sorted(ACTIVITY_MAP.keys()))}
IDX_TO_ACTIVITY = {idx: act for act, idx in ACTIVITY_TO_IDX.items()}

# All subjects in the dataset (ordered)
ALL_SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 108, 109]

# Fixed splits based on plan
TRAIN_SUBJECTS = [101, 102, 103, 104, 108, 109]
VAL_SUBJECTS = [105, 106]
TEST_SUBJECTS = [107]

# Metadata columns used in windowing
META_COLS = ["window_id", "subject_id", "activity_id", "activity_name", "timestep"]
