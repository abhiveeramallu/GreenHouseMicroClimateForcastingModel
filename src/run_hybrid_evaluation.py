"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Run full pipeline and print hybrid model evaluation artifacts.
Module Group: Visualization & Reporting Module (Execution helper)
DFD Connection: Uses pipeline outputs to expose model comparison/ranking and
plant-growth intelligence artifacts for academic report generation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline_data_prep import run_full_project_workflow


def main() -> None:
    results = run_full_project_workflow()

    ranking_path = Path(results["overall_model_ranking"])
    comparison_path = Path(results["overall_model_comparison"])

    print("Hybrid evaluation run completed.")
    print(f"Crops processed: {results['crops_processed']}")
    print(f"Overall results: {results['results_table']}")
    print(f"Model comparison table: {comparison_path}")
    print(f"Model ranking table: {ranking_path}")
    print(f"Architecture doc: {results['architecture_doc']}")
    print(f"Pipeline doc: {results['pipeline_doc']}")

    if ranking_path.exists():
        ranking_df = pd.read_csv(ranking_path)
        if ranking_df.empty:
            print("No model ranking rows available.")
        else:
            print("Top ranked models (average across crops):")
            print(ranking_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
