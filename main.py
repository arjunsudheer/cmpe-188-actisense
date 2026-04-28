from pathlib import Path
from src.preprocessing import dataset_extractor
from src.models import random_forest, logistic_regression, lstm, gru
from src.models.utils.generate_results_table import generate_results_table


def run_pipeline():
    # Preprocess datasets
    processed_base = Path("PAMAP2_Dataset/processed")
    results_base = Path("results")

    dataset_extractor.run_pipeline_step(None, "normal")

    dataset_extractor.run_pipeline_step(
        dataset_extractor.MANUAL_FEATURES_TO_DROP, "feature_selection"
    )

    # Train models and evaluate
    for subdir in ["normal", "feature_selection"]:
        data_dir = processed_base / subdir
        print(f"\nTraining on {subdir} data")

        print("\nRandom Forest")
        random_forest.run_rf(data_dir, results_base / "rf" / subdir)

        print("\nLogistic Regression")
        logistic_regression.run_logreg(data_dir, results_base / "logreg" / subdir)

        print("\nLSTM")
        lstm.run_lstm(data_dir, results_base / "lstm" / subdir)

        print("\nGRU")
        gru.run_gru(data_dir, results_base / "gru" / subdir)

    # Create results table to summarize all model performance
    generate_results_table()


if __name__ == "__main__":
    run_pipeline()
