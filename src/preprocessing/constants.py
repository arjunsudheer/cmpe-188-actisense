# Full set of 18 activities performed across Protocol and Optional sessions
ACTIVITY_MAP = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic walking",
    9: "watching TV",  # Optional – subject 101 only
    10: "computer work",  # Optional – subjects 105, 106, 108, 109
    11: "car driving",  # Optional – subject 101 only
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    18: "folding laundry",  # Optional – subjects 101, 106, 108, 109
    19: "house cleaning",  # Optional – subjects 101, 105, 106, 108, 109
    20: "playing soccer",  # Optional – subjects 108, 109
    24: "rope jumping",
}

# Label mapping for models
ACTIVITY_TO_IDX = {
    1: 0,  # lying
    2: 1,  # sitting
    3: 2,  # standing
    4: 3,  # walking
    5: 4,  # running
    6: 5,  # cycling
    7: 6,  # Nordic walking
    9: 7,  # watching TV
    10: 8,  # computer work
    11: 9,  # car driving
    12: 10,  # ascending stairs
    13: 11,  # descending stairs
    16: 12,  # vacuum cleaning
    17: 13,  # ironing
    18: 14,  # folding laundry
    19: 15,  # house cleaning
    20: 16,  # playing soccer
    24: 17,  # rope jumping
}
IDX_TO_ACTIVITY = {idx: act for act, idx in ACTIVITY_TO_IDX.items()}

# Subject splits for evaluation
TRAIN_SUBJECTS = [101, 102, 103, 104, 105, 109]
VAL_SUBJECTS = [106, 107]
TEST_SUBJECTS = [108]

# Metadata columns used in windowing
META_COLS = ["window_id", "subject_id", "activity_id", "activity_name", "timestep"]
