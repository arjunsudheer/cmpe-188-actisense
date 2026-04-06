from pathlib import Path
from src.preprocessing import dataset_extractor, time_series_formatter
from src.models import random_forest, logistic_regression

def run_pipeline():
    processed_base = Path("PAMAP2_Dataset/processed")
    results_base = Path("results")
    
    # Preprocessing
    df_normal = dataset_extractor.run_pipeline_step(None, "normal")
    df_fs = dataset_extractor.run_pipeline_step(dataset_extractor.MANUAL_FEATURES_TO_DROP, "feature_selection")

    # Windowing for time-series
    time_series_formatter.run_windowing(df_normal, "normal")
    time_series_formatter.run_windowing(df_fs, "feature_selection")

    # Model Training & Evaluation
    for subdir in ["normal", "feature_selection"]:
        random_forest.run_rf(processed_base / subdir, results_base / "rf" / subdir)
        logistic_regression.run_logreg(processed_base / subdir, results_base / "logreg" / subdir)

if __name__ == "__main__":
    run_pipeline()
