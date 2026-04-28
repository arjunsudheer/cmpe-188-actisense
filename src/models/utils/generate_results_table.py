import re
from pathlib import Path
import pandas as pd


def parse_report(report_path):
    if not report_path.exists():
        return None

    content = report_path.read_text()
    results = {}

    # Extract Model Name
    model_match = re.search(r"Metrics Report — (.*)", content)
    results["Model"] = model_match.group(1).strip() if model_match else "Unknown"

    # Extract Metrics for each split
    for split in ["TRAIN", "VAL", "TEST"]:
        pattern = rf"\[{split}\]\n  Accuracy  : ([\d\.]+)\n  Precision : ([\d\.]+)\n  Recall    : ([\d\.]+)\n  F1-Score  : ([\d\.]+)"
        match = re.search(pattern, content)
        if match:
            results[f"{split} Acc"] = float(match.group(1))
            results[f"{split} F1"] = float(match.group(4))

    # Extract Latency
    latency_match = re.search(r"Avg time/sample   : ([\d\.]+) ms", content)
    if latency_match:
        results["Latency (ms)"] = float(latency_match.group(1))

    return results


def generate_results_table():
    results_base = Path("results")
    data = []

    for model_dir in results_base.iterdir():
        if not model_dir.is_dir():
            continue

        for subdir in ["normal", "feature_selection"]:
            report_path = model_dir / subdir / "metrics_report.txt"
            res = parse_report(report_path)
            if res:
                res["Config"] = subdir
                data.append(res)

    if not data:
        print("No reports found.")
        return

    df = pd.DataFrame(data)

    # Reorder columns
    cols = [
        "Model",
        "Config",
        "TRAIN Acc",
        "VAL Acc",
        "TEST Acc",
        "TEST F1",
        "Latency (ms)",
    ]
    df = df[[c for c in cols if c in df.columns]]

    # Sort by TEST F1-Score
    if "TEST F1" in df.columns:
        df = df.sort_values("TEST F1", ascending=False)
    else:
        df = df.sort_values("TEST Acc", ascending=False)

    table_md = "# PAMAP2 Model Comparison Table\n\n" + df.to_markdown(index=False)
    
    # Save to file
    table_path = results_base / "summary_table.md"
    table_path.write_text(table_md)
    print(f" Saved results table to {table_path}")

    # Also print to console
    print("\n" + table_md)
