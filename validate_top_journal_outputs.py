"""
Validate required outputs for the cBD3-ABU top-journal dry-lab pipeline.

Run after:

    python deep_evo_top_journal_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUT_DIR = Path("results/top_journal")
REQUIRED_FEATURE_SETS = {"DescriptorOnly", "ESM2Only", "DescriptorPlusESM2"}
REQUIRED_MODELS = {"LogisticRegression", "RandomForest", "HistGradientBoosting"}
REQUIRED_OBJECTIVES = {
    "PredictedActivity",
    "SafetyNoCys",
    "Amphipathicity",
    "ChargeSuitability",
    "ScaffoldSimilarity",
    "Synthesizability",
    "HydrophobicRunSafety",
    "HydrophobicExcessSafety",
}
REQUIRED_EVIDENCE = {
    "RMSD_80_100_mean",
    "RMSF_mean",
    "MIC_numeric_D37-2",
    "MIC_numeric_D84-1",
    "HemolysisAtMaxConcentration",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    require(OUT_DIR.exists(), f"missing output directory: {OUT_DIR}")

    audit = json.loads((OUT_DIR / "dataset_audit.json").read_text(encoding="utf-8"))
    require(audit["raw_rows"] == 570, "raw peptide count must be 570")
    require(audit["labeled_rows"] == 171, "labeled training set must be 171")
    require(audit["positive_label_count"] == 88, "positive count must be 88")
    require(audit["negative_label_count"] == 83, "negative count must be 83")
    require(audit["anchor_present"], "cBD3-ABU anchor must be present")

    embedding_status = (OUT_DIR / "embedding_status.txt").read_text(encoding="utf-8").strip()
    require(
        embedding_status.startswith(("esm2:", "cache:")),
        f"ESM-2 status must be real or cached ESM-2, got {embedding_status!r}",
    )

    metrics = pd.read_csv(OUT_DIR / "model_cv_metrics_summary.csv")
    require(set(metrics["FeatureSet"]) == REQUIRED_FEATURE_SETS, "missing model feature-set ablation")
    require(set(metrics["Model"]) == REQUIRED_MODELS, "missing model-family ablation")
    require(metrics["ROC_AUC_mean"].notna().all(), "ROC-AUC summary contains NaN")
    require(metrics["F1_mean"].notna().all(), "F1 summary contains NaN")

    pareto = pd.read_csv(OUT_DIR / "candidate_pareto_nsga2.csv")
    require(REQUIRED_OBJECTIVES.issubset(pareto.columns), "missing Pareto objective columns")
    require({"ParetoRank", "CrowdingDistance"}.issubset(pareto.columns), "missing NSGA-II columns")
    require((pareto["ParetoRank"] >= 1).all(), "Pareto rank must start at 1")

    evidence = pd.read_csv(OUT_DIR / "candidate_integrated_evidence.csv")
    require(set(["Rank7", "Rank9", "Rank12", "cBD3-ABU"]).issubset(evidence["Peptide"]), "missing lead peptides")
    require(REQUIRED_EVIDENCE.issubset(evidence.columns), "missing MD, MIC, or hemolysis evidence")

    print("Top-journal pipeline outputs validated.")
    print(f"Embedding status: {embedding_status}")
    print(f"Best model: {metrics.iloc[0]['FeatureSet']} + {metrics.iloc[0]['Model']}")


if __name__ == "__main__":
    main()
