"""
Top-journal dry-lab pipeline for cBD3-ABU scaffold rescue.

This script turns the existing DeepEvo-Pept prototype into a reproducible
evidence generator:

1. audit the raw and labeled peptide datasets
2. compute interpretable descriptors and real ESM-2 embeddings
3. benchmark descriptor-only, ESM2-only, and hybrid predictors
4. score constrained-GA candidates
5. apply NSGA-II-style Pareto ranking
6. integrate candidate scores with MD, MIC, and hemolysis evidence
7. generate publication-oriented tables and figures under results/

CPU is supported. The default ESM-2 model is the small
facebook/esm2_t6_8M_UR50D checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
ANCHOR_SEQ = "KAWNLRGSAREKAIKNEKLYIFATSGKLAALKPK"
RANK_SEQS = {
    "Rank7": "SVWQLQKWARPVALKLDYLYQFATLGSLAKLKPK",
    "Rank9": "YAWTLNKLNFNKALTIDPLKNVYKLKKLAKLKWK",
    "Rank12": "SAWQLYGSARPKALKNDYLYIFATFGQLAKLKPK",
}

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_STRONG = set("LVIFW")
POLAR_CHARGED = set("KRDEQNSTHY")
KD_HYDRO = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}
HELIX_PROP = {
    "A": 1.45,
    "R": 0.79,
    "N": 0.73,
    "D": 0.98,
    "C": 0.77,
    "Q": 1.17,
    "E": 1.53,
    "G": 0.53,
    "H": 1.00,
    "I": 1.00,
    "L": 1.34,
    "K": 1.07,
    "M": 1.20,
    "F": 1.12,
    "P": 0.59,
    "S": 0.79,
    "T": 0.82,
    "W": 1.14,
    "Y": 0.61,
    "V": 1.14,
}


def set_reproducible(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    warnings.filterwarnings(
        "ignore",
        message=".*encountered in matmul",
        category=RuntimeWarning,
        module="sklearn.utils.extmath",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_sequence(seq: str) -> str:
    return "".join(ch for ch in str(seq).upper().strip() if ch.isalpha())


def extract_numeric(value) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    import re

    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return float("nan")
    return float(match.group(0))


def compute_aac(seq: str) -> List[float]:
    counts = Counter(seq)
    return [counts.get(aa, 0) / max(1, len(seq)) for aa in AA_LIST]


def compute_dipeptide_freq(seq: str) -> List[float]:
    pairs = [seq[i : i + 2] for i in range(max(0, len(seq) - 1))]
    counts = Counter(pairs)
    values = []
    denom = max(1, len(pairs))
    for a in AA_LIST:
        for b in AA_LIST:
            values.append(counts.get(a + b, 0) / denom)
    return values


def hydrophobic_fraction(seq: str) -> float:
    return sum(aa in HYDROPHOBIC_STRONG for aa in seq) / max(1, len(seq))


def longest_hydrophobic_run(seq: str) -> int:
    longest = 0
    current = 0
    for aa in seq:
        if aa in HYDROPHOBIC_STRONG:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def net_charge(seq: str) -> int:
    return seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")


def hydrophobic_moment(seq: str) -> float:
    theta = math.radians(100)
    sx = 0.0
    sy = 0.0
    for i, aa in enumerate(seq):
        h = KD_HYDRO.get(aa, 0.0)
        sx += h * math.cos(i * theta)
        sy += h * math.sin(i * theta)
    return math.sqrt(sx * sx + sy * sy) / max(1, len(seq))


def helix_propensity(seq: str) -> Tuple[float, float]:
    values = np.array([HELIX_PROP.get(aa, 0.0) for aa in seq], dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    window = 9
    if len(values) < window:
        max_window = float(values.mean())
    else:
        max_window = float(max(values[i : i + window].mean() for i in range(len(values) - window + 1)))
    return float(values.mean()), max_window


def alternation_score(seq: str) -> float:
    classes = []
    for aa in seq:
        if aa in HYDROPHOBIC_STRONG:
            classes.append("H")
        elif aa in POLAR_CHARGED:
            classes.append("P")
        else:
            classes.append("N")
    switches = 0
    comparable = 0
    for a, b in zip(classes, classes[1:]):
        if "N" in (a, b):
            continue
        comparable += 1
        switches += a != b
    return switches / max(1, comparable)


def molecular_weight_rough(seq: str) -> float:
    masses = {
        "A": 89.09,
        "R": 174.20,
        "N": 132.12,
        "D": 133.10,
        "C": 121.15,
        "E": 147.13,
        "Q": 146.15,
        "G": 75.07,
        "H": 155.16,
        "I": 131.17,
        "L": 131.17,
        "K": 146.19,
        "M": 149.21,
        "F": 165.19,
        "P": 115.13,
        "S": 105.09,
        "T": 119.12,
        "W": 204.23,
        "Y": 181.19,
        "V": 117.15,
    }
    if not seq:
        return 0.0
    return sum(masses.get(aa, 0.0) for aa in seq) - (len(seq) - 1) * 18.015


def descriptor_features(seq: str) -> np.ndarray:
    seq = clean_sequence(seq)
    kd_values = np.array([KD_HYDRO.get(aa, 0.0) for aa in seq], dtype=float)
    h_mean, h_win = helix_propensity(seq)
    positive_fraction = (seq.count("K") + seq.count("R")) / max(1, len(seq))
    negative_fraction = (seq.count("D") + seq.count("E")) / max(1, len(seq))
    core = [
        len(seq),
        float(kd_values.mean()) if len(kd_values) else 0.0,
        float(kd_values.std()) if len(kd_values) else 0.0,
        hydrophobic_fraction(seq),
        hydrophobic_moment(seq),
        h_mean,
        h_win,
        molecular_weight_rough(seq),
        net_charge(seq),
        positive_fraction,
        negative_fraction,
        alternation_score(seq),
        longest_hydrophobic_run(seq),
        1.0 if "C" in seq else 0.0,
    ]
    return np.array(compute_aac(seq) + compute_dipeptide_freq(seq) + core, dtype=float)


def feature_names() -> List[str]:
    names = [f"AAC_{aa}" for aa in AA_LIST]
    names += [f"DIPEP_{a}{b}" for a in AA_LIST for b in AA_LIST]
    names += [
        "Length",
        "KD_mean",
        "KD_std",
        "StrongHydrophobicFraction",
        "HydrophobicMoment",
        "HelixMean",
        "HelixMaxWindow9",
        "RoughMolecularWeight",
        "NetCharge",
        "PositiveFraction",
        "NegativeFraction",
        "AlternationScore",
        "LongestHydrophobicRun",
        "HasCys",
    ]
    return names


def sequence_identity(a: str, b: str) -> float:
    a = clean_sequence(a)
    b = clean_sequence(b)
    if not a or not b:
        return 0.0
    n = max(len(a), len(b))
    matches = sum(x == y for x, y in zip(a, b))
    return matches / n


def levenshtein(a: str, b: str) -> int:
    a = clean_sequence(a)
    b = clean_sequence(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_or_build_labeled_data(raw_path: Path, labeled_path: Path, out_dir: Path) -> pd.DataFrame:
    if not labeled_path.exists():
        if not raw_path.exists():
            raise FileNotFoundError("final_train_set.csv and my_500_peptides.csv are both missing")
        raw = pd.read_csv(raw_path)
        mic_col = next((c for c in raw.columns if "MIC" in c), "MIC (µg/M)")
        hem_col = next((c for c in raw.columns if "Hemolysis" in c and "%" in c), "Hemolysis (%)")
        raw[mic_col] = raw[mic_col].map(extract_numeric)
        raw[hem_col] = raw[hem_col].map(extract_numeric)
        raw = raw.dropna(subset=["Sequence", mic_col])
        raw["Label"] = np.nan
        raw.loc[(raw[mic_col] <= 10) & (raw[hem_col] < 20), "Label"] = 1
        raw.loc[raw[mic_col] >= 32, "Label"] = 0
        labeled = raw.dropna(subset=["Label"]).copy()
        labeled["Label"] = labeled["Label"].astype(int)
        if ANCHOR_SEQ not in set(labeled["Sequence"].astype(str)):
            anchor = {col: None for col in labeled.columns}
            anchor.update({"Sequence": ANCHOR_SEQ, mic_col: 64, hem_col: 5, "Label": 0})
            labeled = pd.concat([labeled, pd.DataFrame([anchor])], ignore_index=True)
        labeled.to_csv(labeled_path, index=False)
    labeled = pd.read_csv(labeled_path)
    labeled["Sequence"] = labeled["Sequence"].map(clean_sequence)
    labeled["Label"] = labeled["Label"].astype(int)
    audit = {
        "raw_rows": int(pd.read_csv(raw_path).shape[0]) if raw_path.exists() else None,
        "labeled_rows": int(labeled.shape[0]),
        "positive_label_count": int((labeled["Label"] == 1).sum()),
        "negative_label_count": int((labeled["Label"] == 0).sum()),
        "anchor_present": bool((labeled["Sequence"] == ANCHOR_SEQ).any()),
    }
    (out_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame([audit]).to_csv(out_dir / "dataset_audit.csv", index=False)
    return labeled


def compute_esm2_embeddings(sequences: Sequence[str], model_name: str, cache_path: Path, batch_size: int = 8) -> Tuple[np.ndarray, str]:
    sequences = [clean_sequence(s) for s in sequences]
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_sequences = cached["sequences"].tolist()
        if cached_sequences == sequences and str(cached["model_name"]) == model_name:
            return cached["embeddings"], f"cache:{model_name}"

    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            toks = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True)
            out = model(**toks)
            hidden = out.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.cpu().numpy())
    arr = np.vstack(embeddings)
    np.savez_compressed(cache_path, sequences=np.array(sequences, dtype=object), embeddings=arr, model_name=model_name)
    return arr, f"esm2:{model_name}"


def evaluate_models(
    X_sets: Dict[str, np.ndarray],
    y: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    models = {
        "LogisticRegression": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="liblinear",
                        C=0.5,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            random_state=SEED,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for feature_set, X in X_sets.items():
        for model_name, factory in models.items():
            fold = 0
            for train_idx, test_idx in cv.split(X, y):
                fold += 1
                model = factory()
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X[test_idx])[:, 1]
                else:
                    prob = pred.astype(float)
                rows.append(
                    {
                        "FeatureSet": feature_set,
                        "Model": model_name,
                        "Fold": fold,
                        "Accuracy": accuracy_score(y[test_idx], pred),
                        "ROC_AUC": roc_auc_score(y[test_idx], prob),
                        "Precision": precision_score(y[test_idx], pred, zero_division=0),
                        "Recall": recall_score(y[test_idx], pred, zero_division=0),
                        "F1": f1_score(y[test_idx], pred, zero_division=0),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "model_cv_metrics_by_fold.csv", index=False)
    summary = (
        metrics.groupby(["FeatureSet", "Model"], as_index=False)
        .agg({m: ["mean", "std"] for m in ["Accuracy", "ROC_AUC", "Precision", "Recall", "F1"]})
    )
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns]
    summary = summary.sort_values(["ROC_AUC_mean", "F1_mean"], ascending=False)
    summary.to_csv(out_dir / "model_cv_metrics_summary.csv", index=False)
    return summary


def fit_selected_model(feature_set: str, model_name: str, X_sets: Dict[str, np.ndarray], y: np.ndarray):
    X = X_sets[feature_set]
    if model_name == "LogisticRegression":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="liblinear",
                        C=0.5,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    elif model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, random_state=SEED)
    model.fit(X, y)
    return model


@dataclass
class CandidateRecord:
    source_rank: int
    source_score: float
    gen_found: int
    sequence: str
    objectives: List[float]
    pareto_rank: int = 0
    crowding_distance: float = 0.0


def dominates(a: CandidateRecord, b: CandidateRecord) -> bool:
    return all(x >= y for x, y in zip(a.objectives, b.objectives)) and any(
        x > y for x, y in zip(a.objectives, b.objectives)
    )


def non_dominated_sort(cands: List[CandidateRecord]) -> List[List[CandidateRecord]]:
    fronts: List[List[CandidateRecord]] = []
    domination_counts = {id(c): 0 for c in cands}
    dominated = {id(c): [] for c in cands}
    first = []
    for p in cands:
        for q in cands:
            if p is q:
                continue
            if dominates(p, q):
                dominated[id(p)].append(q)
            elif dominates(q, p):
                domination_counts[id(p)] += 1
        if domination_counts[id(p)] == 0:
            p.pareto_rank = 1
            first.append(p)
    fronts.append(first)
    rank = 1
    while fronts[-1]:
        nxt = []
        for p in fronts[-1]:
            for q in dominated[id(p)]:
                domination_counts[id(q)] -= 1
                if domination_counts[id(q)] == 0:
                    q.pareto_rank = rank + 1
                    nxt.append(q)
        rank += 1
        fronts.append(nxt)
    return fronts[:-1]


def assign_crowding(front: List[CandidateRecord]) -> None:
    if not front:
        return
    for c in front:
        c.crowding_distance = 0.0
    n_obj = len(front[0].objectives)
    for j in range(n_obj):
        front.sort(key=lambda c: c.objectives[j])
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        lo = front[0].objectives[j]
        hi = front[-1].objectives[j]
        if hi == lo:
            continue
        for i in range(1, len(front) - 1):
            front[i].crowding_distance += (front[i + 1].objectives[j] - front[i - 1].objectives[j]) / (hi - lo)


def candidate_objectives(
    sequence: str,
    predicted_activity: float,
    candidate_embedding: np.ndarray,
    anchor_embedding: np.ndarray,
) -> List[float]:
    safety = 1.0
    if "C" in sequence:
        safety -= 0.5
    safety -= min(0.5, max(0, longest_hydrophobic_run(sequence) - 2) / 8.0)
    safety -= min(0.5, max(0.0, hydrophobic_fraction(sequence) - 0.50) * 2.0)
    safety = max(0.0, safety)

    charge_score = 1.0 - min(1.0, abs(net_charge(sequence) - 7) / 12.0)
    scaffold_similarity = cosine_similarity(candidate_embedding, anchor_embedding)
    synthesizability = 1.0
    synthesizability -= 0.4 if "C" in sequence else 0.0
    synthesizability -= min(0.4, max(0.0, hydrophobic_fraction(sequence) - 0.50) * 1.5)
    synthesizability -= min(0.2, max(0, longest_hydrophobic_run(sequence) - 2) / 10.0)
    synthesizability = max(0.0, synthesizability)
    hydrophobic_run_safety = 1.0 - min(1.0, max(0, longest_hydrophobic_run(sequence) - 2) / 6.0)
    hydro_excess_safety = 1.0 - min(1.0, max(0.0, hydrophobic_fraction(sequence) - 0.50) * 4.0)
    amphipathicity = 0.5 * hydrophobic_moment(sequence) + 0.5 * alternation_score(sequence)
    return [
        predicted_activity,
        safety,
        amphipathicity,
        charge_score,
        scaffold_similarity,
        synthesizability,
        hydrophobic_run_safety,
        hydro_excess_safety,
    ]


def run_pareto(
    candidates: pd.DataFrame,
    predicted_scores: np.ndarray,
    candidate_embeddings: np.ndarray,
    anchor_embedding: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    records = []
    for i, row in candidates.iterrows():
        records.append(
            CandidateRecord(
                source_rank=int(row["Rank"]),
                source_score=float(row["Score"]),
                gen_found=int(row["GenFound"]),
                sequence=row["Sequence"],
                objectives=candidate_objectives(
                    row["Sequence"],
                    float(predicted_scores[i]),
                    candidate_embeddings[i],
                    anchor_embedding,
                ),
            )
        )
    fronts = non_dominated_sort(records)
    for front in fronts:
        assign_crowding(front)
    rows = []
    obj_names = [
        "PredictedActivity",
        "SafetyNoCys",
        "Amphipathicity",
        "ChargeSuitability",
        "ScaffoldSimilarity",
        "Synthesizability",
        "HydrophobicRunSafety",
        "HydrophobicExcessSafety",
    ]
    for r in sorted(records, key=lambda x: (x.pareto_rank, -x.crowding_distance, x.source_rank)):
        item = {
            "ParetoRank": r.pareto_rank,
            "CrowdingDistance": r.crowding_distance,
            "SourceRank": r.source_rank,
            "SourceScore": r.source_score,
            "GenFound": r.gen_found,
            "Sequence": r.sequence,
            "NetCharge": net_charge(r.sequence),
            "StrongHydrophobicFraction": hydrophobic_fraction(r.sequence),
            "LongestHydrophobicRun": longest_hydrophobic_run(r.sequence),
            "HydrophobicMoment": hydrophobic_moment(r.sequence),
            "AlternationScore": alternation_score(r.sequence),
        }
        item.update(dict(zip(obj_names, r.objectives)))
        rows.append(item)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "candidate_pareto_nsga2.csv", index=False)
    return out


def nearest_neighbors(train: pd.DataFrame, candidates: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    train_sequences = train["Sequence"].tolist()
    for _, c in candidates.iterrows():
        best_seq = None
        best_identity = -1.0
        best_distance = 10**9
        for seq in train_sequences:
            ident = sequence_identity(c["Sequence"], seq)
            dist = levenshtein(c["Sequence"], seq)
            if ident > best_identity or (ident == best_identity and dist < best_distance):
                best_identity = ident
                best_distance = dist
                best_seq = seq
        rows.append(
            {
                "CandidateRank": c["Rank"],
                "CandidateSequence": c["Sequence"],
                "AnchorIdentity": sequence_identity(c["Sequence"], ANCHOR_SEQ),
                "AnchorLevenshtein": levenshtein(c["Sequence"], ANCHOR_SEQ),
                "NearestTrainingSequence": best_seq,
                "NearestTrainingIdentity": best_identity,
                "NearestTrainingLevenshtein": best_distance,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "candidate_similarity_nearest_neighbors.csv", index=False)
    return out


def load_md_summary(md_dir: Path) -> pd.DataFrame:
    mapping = {
        "cBD3-ABU": ("cbd3_abu-rmsd.csv", "cbd3_abu-rmsf.csv"),
        "Rank7": ("rank_7-rmsd.csv", "rank_7-rmsf.csv"),
        "Rank9": ("rank_9-rmsd.csv", "rank_9-rmsf.csv"),
        "Rank12": ("rank_12-rmsd.csv", "rank_12-rmsf.csv"),
    }
    rows = []
    for name, (rmsd_file, rmsf_file) in mapping.items():
        rmsd_path = md_dir / rmsd_file
        rmsf_path = md_dir / rmsf_file
        if not rmsd_path.exists() or not rmsf_path.exists():
            continue
        rmsd = pd.read_csv(rmsd_path)
        rmsf = pd.read_csv(rmsf_path)
        rmsd_col = [c for c in rmsd.columns if "RMSD" in c][0]
        rmsf_col = [c for c in rmsf.columns if "RMSF" in c][0]
        tail = rmsd[rmsd["Time (ns)"] >= 80][rmsd_col]
        rows.append(
            {
                "Peptide": name,
                "RMSD_final": float(rmsd[rmsd_col].iloc[-1]),
                "RMSD_mean": float(rmsd[rmsd_col].mean()),
                "RMSD_80_100_mean": float(tail.mean()),
                "RMSD_80_100_sd": float(tail.std(ddof=1)),
                "RMSF_mean": float(rmsf[rmsf_col].mean()),
                "RMSF_max": float(rmsf[rmsf_col].max()),
            }
        )
    return pd.DataFrame(rows)


def load_mic(mic_path: Path) -> pd.DataFrame:
    if not mic_path.exists():
        return pd.DataFrame()
    raw = pd.read_excel(mic_path)
    raw = raw.rename(columns={raw.columns[0]: "Peptide"})
    rows = []
    aliases = {"CBD3-ABU": "cBD3-ABU", "Rank7": "Rank7", "Rank9": "Rank9", "Rank12": "Rank12"}
    for _, row in raw.iterrows():
        pep = str(row["Peptide"]).strip()
        if pep not in aliases:
            continue
        values = {}
        for col in raw.columns[1:]:
            val = row[col]
            values[f"MIC_{col}"] = val
            values[f"MIC_numeric_{col}"] = 512.0 if str(val).strip() in {"耐药", ">256"} else extract_numeric(val)
        values["Peptide"] = aliases[pep]
        rows.append(values)
    return pd.DataFrame(rows)


def load_hemolysis(hemo_path: Path) -> pd.DataFrame:
    if not hemo_path.exists():
        return pd.DataFrame()
    sheet = pd.read_excel(hemo_path, sheet_name="Sheet1", header=None)
    rows = []
    current = None
    aliases = {"CBD3-ABU": "cBD3-ABU", "Rank 7": "Rank7", "Rank 9": "Rank9", "Rank 12": "Rank12"}
    for _, row in sheet.iterrows():
        first = row.iloc[0]
        if isinstance(first, str) and first.strip() in aliases:
            current = aliases[first.strip()]
            continue
        if current and pd.notna(first) and str(first).strip().replace(".", "", 1).isdigit():
            rows.append(
                {
                    "Peptide": current,
                    "HemolysisConcentration": float(first),
                    "HemolysisMean": float(row.iloc[1]),
                    "HemolysisSD": float(row.iloc[2]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    max_rows = out.sort_values("HemolysisConcentration").groupby("Peptide", as_index=False).tail(1)
    max_rows = max_rows.rename(
        columns={
            "HemolysisConcentration": "HemolysisMaxConcentration",
            "HemolysisMean": "HemolysisAtMaxConcentration",
            "HemolysisSD": "HemolysisSDAtMaxConcentration",
        }
    )
    return max_rows


def build_integrated_evidence(
    pareto: pd.DataFrame,
    md: pd.DataFrame,
    mic: pd.DataFrame,
    hemo: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    selected = []
    for name, seq in {"Rank7": RANK_SEQS["Rank7"], "Rank9": RANK_SEQS["Rank9"], "Rank12": RANK_SEQS["Rank12"]}.items():
        match = pareto[pareto["Sequence"] == seq].copy()
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        row["Peptide"] = name
        selected.append(row)
    cbd3 = {
        "Peptide": "cBD3-ABU",
        "Sequence": ANCHOR_SEQ,
        "NetCharge": net_charge(ANCHOR_SEQ),
        "StrongHydrophobicFraction": hydrophobic_fraction(ANCHOR_SEQ),
        "LongestHydrophobicRun": longest_hydrophobic_run(ANCHOR_SEQ),
        "HydrophobicMoment": hydrophobic_moment(ANCHOR_SEQ),
        "AlternationScore": alternation_score(ANCHOR_SEQ),
    }
    selected.append(cbd3)
    out = pd.DataFrame(selected)
    for table in [md, mic, hemo]:
        if not table.empty:
            out = out.merge(table, on="Peptide", how="left")
    out.to_csv(out_dir / "candidate_integrated_evidence.csv", index=False)
    return out


def random_mutation_baseline(seed_seq: str, n: int = 500) -> pd.DataFrame:
    rng = random.Random(SEED)
    rows = []
    for i in range(n):
        seq = list(seed_seq)
        points = rng.randint(1, 5)
        for _ in range(points):
            pos = rng.randrange(len(seq))
            choices = [aa for aa in AA_LIST if aa != seq[pos]]
            seq[pos] = rng.choice(choices)
        s = "".join(seq)
        rows.append(
            {
                "Method": "RandomMutation",
                "Sequence": s,
                "CysFree": "C" not in s,
                "HydrophobicRunViolation": longest_hydrophobic_run(s) >= 3,
                "StrongHydrophobicFraction": hydrophobic_fraction(s),
                "NetCharge": net_charge(s),
                "HydrophobicMoment": hydrophobic_moment(s),
            }
        )
    return pd.DataFrame(rows)


def ablation_table(candidates_basic: pd.DataFrame, candidates_v2: pd.DataFrame, pareto: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    tables = []
    random_df = random_mutation_baseline(ANCHOR_SEQ)
    tables.append(random_df)
    for method, df in [("BasicGA", candidates_basic), ("ConstrainedGA", candidates_v2)]:
        rows = []
        for _, row in df.iterrows():
            s = row["Sequence"]
            rows.append(
                {
                    "Method": method,
                    "Sequence": s,
                    "CysFree": "C" not in s,
                    "HydrophobicRunViolation": longest_hydrophobic_run(s) >= 3,
                    "StrongHydrophobicFraction": hydrophobic_fraction(s),
                    "NetCharge": net_charge(s),
                    "HydrophobicMoment": hydrophobic_moment(s),
                    "TopScore": row.get("Score", np.nan),
                }
            )
        tables.append(pd.DataFrame(rows))
    nsga = pareto.copy()
    nsga_rows = []
    for _, row in nsga.iterrows():
        s = row["Sequence"]
        nsga_rows.append(
            {
                "Method": "ESM2_NSGA2",
                "Sequence": s,
                "CysFree": "C" not in s,
                "HydrophobicRunViolation": longest_hydrophobic_run(s) >= 3,
                "StrongHydrophobicFraction": hydrophobic_fraction(s),
                "NetCharge": net_charge(s),
                "HydrophobicMoment": hydrophobic_moment(s),
                "TopScore": row.get("PredictedActivity", np.nan),
                "ParetoRank": row.get("ParetoRank", np.nan),
            }
        )
    tables.append(pd.DataFrame(nsga_rows))
    all_rows = pd.concat(tables, ignore_index=True)
    summary = (
        all_rows.groupby("Method", as_index=False)
        .agg(
            CandidateCount=("Sequence", "count"),
            CysFreeRate=("CysFree", "mean"),
            HydrophobicRunViolationRate=("HydrophobicRunViolation", "mean"),
            MeanStrongHydrophobicFraction=("StrongHydrophobicFraction", "mean"),
            MeanNetCharge=("NetCharge", "mean"),
            MeanHydrophobicMoment=("HydrophobicMoment", "mean"),
            MeanTopScore=("TopScore", "mean"),
            BestTopScore=("TopScore", "max"),
            MeanParetoRank=("ParetoRank", "mean"),
        )
        .sort_values("Method")
    )
    all_rows.to_csv(out_dir / "ablation_candidate_level.csv", index=False)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    return summary


def make_figures(
    train: pd.DataFrame,
    x_desc: np.ndarray,
    emb_all: np.ndarray,
    sequence_index: Dict[str, int],
    candidates: pd.DataFrame,
    pareto: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    integrated: pd.DataFrame,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig_dir = ensure_dir(out_dir / "figures")

    # Figure 1: model metrics heatmap-like bar plot.
    top = metrics_summary.copy()
    top["Label"] = top["FeatureSet"] + "\n" + top["Model"]
    plt.figure(figsize=(10, 4.8))
    order = np.arange(len(top))
    plt.bar(order, top["ROC_AUC_mean"], label="ROC-AUC", alpha=0.85)
    plt.bar(order, top["F1_mean"], label="F1", alpha=0.65)
    plt.xticks(order, top["Label"], rotation=55, ha="right", fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("5-fold CV score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_model_ablation.png", dpi=220)
    plt.close()

    # Figure 2: PCA of ESM embeddings.
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(emb_all)
    labels = []
    seqs = list(sequence_index.keys())
    candidate_set = set(candidates["Sequence"])
    for seq in seqs:
        if seq == ANCHOR_SEQ:
            labels.append("cBD3-ABU")
        elif seq in set(RANK_SEQS.values()):
            labels.append({v: k for k, v in RANK_SEQS.items()}[seq])
        elif seq in candidate_set:
            labels.append("Candidates")
        else:
            labels.append("Training")
    pca_df = pd.DataFrame({"Sequence": seqs, "PC1": coords[:, 0], "PC2": coords[:, 1], "Group": labels})
    pca_df.to_csv(out_dir / "esm2_pca_coordinates.csv", index=False)
    colors = {"Training": "#9aa0a6", "Candidates": "#4c78a8", "cBD3-ABU": "#111111", "Rank7": "#2ca02c", "Rank9": "#d62728", "Rank12": "#9467bd"}
    plt.figure(figsize=(7, 5))
    for group, gdf in pca_df.groupby("Group"):
        plt.scatter(gdf["PC1"], gdf["PC2"], s=55 if group.startswith("Rank") or group == "cBD3-ABU" else 18, label=group, alpha=0.85, c=colors.get(group))
    plt.xlabel("ESM-2 PC1")
    plt.ylabel("ESM-2 PC2")
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_esm2_pca.png", dpi=220)
    plt.close()

    # Figure 3: Pareto front.
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        pareto["PredictedActivity"],
        pareto["Amphipathicity"],
        c=pareto["ParetoRank"],
        s=80,
        cmap="viridis_r",
        edgecolors="black",
        linewidths=0.4,
    )
    for name, seq in RANK_SEQS.items():
        row = pareto[pareto["Sequence"] == seq]
        if not row.empty:
            plt.annotate(name, (row["PredictedActivity"].iloc[0], row["Amphipathicity"].iloc[0]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Predicted activity objective")
    plt.ylabel("Amphipathicity objective")
    plt.colorbar(scatter, label="Pareto rank")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_pareto_front.png", dpi=220)
    plt.close()

    # Figure 4: integrated evidence heatmap.
    heat_cols = [
        "PredictedActivity",
        "SafetyNoCys",
        "Amphipathicity",
        "ChargeSuitability",
        "ScaffoldSimilarity",
        "RMSF_mean",
        "RMSD_80_100_mean",
        "HemolysisAtMaxConcentration",
    ]
    available = [c for c in heat_cols if c in integrated.columns]
    if available:
        mat = integrated.set_index("Peptide")[available].apply(pd.to_numeric, errors="coerce")
        norm = (mat - mat.min()) / (mat.max() - mat.min()).replace(0, 1)
        plt.figure(figsize=(9, 4.5))
        plt.imshow(norm.fillna(0), aspect="auto", cmap="mako" if False else "viridis")
        plt.yticks(np.arange(len(norm.index)), norm.index)
        plt.xticks(np.arange(len(norm.columns)), norm.columns, rotation=45, ha="right", fontsize=8)
        plt.colorbar(label="normalized value")
        plt.tight_layout()
        plt.savefig(fig_dir / "figure_integrated_evidence_heatmap.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="my_500_peptides.csv")
    parser.add_argument("--labeled", default="final_train_set.csv")
    parser.add_argument("--candidates-basic", default="candidates.csv")
    parser.add_argument("--candidates-v2", default="candidates_v2.csv")
    parser.add_argument("--out", default="results/top_journal")
    parser.add_argument("--esm2-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--md-dir", default="../../03_计算模拟结果/cBD3-ABU等膜动力学4组/md-results")
    parser.add_argument("--mic", default="../../数据/MIC测定 4.21.xlsx")
    parser.add_argument("--hemolysis", default="../../数据/溶血率 5.24.xlsx")
    args = parser.parse_args()

    set_reproducible()
    out_dir = ensure_dir(Path(args.out))

    train = load_or_build_labeled_data(Path(args.raw), Path(args.labeled), out_dir)
    candidates_v2 = pd.read_csv(args.candidates_v2)
    candidates_v2["Sequence"] = candidates_v2["Sequence"].map(clean_sequence)
    candidates_basic = pd.read_csv(args.candidates_basic) if Path(args.candidates_basic).exists() else pd.DataFrame(columns=candidates_v2.columns)

    train_sequences = train["Sequence"].tolist()
    candidate_sequences = candidates_v2["Sequence"].tolist()
    all_sequences = list(dict.fromkeys(train_sequences + candidate_sequences + [ANCHOR_SEQ] + list(RANK_SEQS.values())))
    sequence_index = {seq: i for i, seq in enumerate(all_sequences)}

    X_desc_all = np.vstack([descriptor_features(seq) for seq in all_sequences])
    pd.DataFrame(X_desc_all, index=all_sequences, columns=feature_names()).to_csv(out_dir / "descriptor_features_all_sequences.csv")

    embeddings, embedding_status = compute_esm2_embeddings(
        all_sequences,
        args.esm2_model,
        out_dir / "esm2_embeddings_all_sequences.npz",
    )
    pd.DataFrame(embeddings, index=all_sequences).to_csv(out_dir / "esm2_embeddings_all_sequences.csv")
    (out_dir / "embedding_status.txt").write_text(embedding_status + "\n", encoding="utf-8")

    train_idx = [sequence_index[s] for s in train_sequences]
    cand_idx = [sequence_index[s] for s in candidate_sequences]
    anchor_idx = sequence_index[ANCHOR_SEQ]

    X_desc_train = X_desc_all[train_idx]
    X_esm_train = embeddings[train_idx]
    X_hybrid_train = np.hstack([X_desc_train, X_esm_train])
    y = train["Label"].to_numpy(dtype=int)

    X_sets = {
        "DescriptorOnly": X_desc_train,
        "ESM2Only": X_esm_train,
        "DescriptorPlusESM2": X_hybrid_train,
    }
    metrics_summary = evaluate_models(X_sets, y, out_dir)
    best = metrics_summary.iloc[0]
    selected_feature_set = str(best["FeatureSet"])
    selected_model_name = str(best["Model"])
    selected_model = fit_selected_model(selected_feature_set, selected_model_name, X_sets, y)
    selection = {
        "selected_feature_set": selected_feature_set,
        "selected_model": selected_model_name,
        "selection_rule": "highest mean ROC-AUC, then highest mean F1",
        "embedding_status": embedding_status,
    }
    (out_dir / "selected_model.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    X_desc_candidates = X_desc_all[cand_idx]
    X_esm_candidates = embeddings[cand_idx]
    if selected_feature_set == "DescriptorOnly":
        X_candidates = X_desc_candidates
    elif selected_feature_set == "ESM2Only":
        X_candidates = X_esm_candidates
    else:
        X_candidates = np.hstack([X_desc_candidates, X_esm_candidates])
    predicted_scores = selected_model.predict_proba(X_candidates)[:, 1]
    candidates_scored = candidates_v2.copy()
    candidates_scored["SelectedModelScore"] = predicted_scores
    candidates_scored.to_csv(out_dir / "candidates_v2_selected_model_scores.csv", index=False)

    nearest_neighbors(train, candidates_v2, out_dir)
    pareto = run_pareto(candidates_v2, predicted_scores, embeddings[cand_idx], embeddings[anchor_idx], out_dir)
    md = load_md_summary(Path(args.md_dir))
    md.to_csv(out_dir / "md_summary.csv", index=False)
    mic = load_mic(Path(args.mic))
    if not mic.empty:
        mic.to_csv(out_dir / "mic_summary.csv", index=False)
    hemo = load_hemolysis(Path(args.hemolysis))
    if not hemo.empty:
        hemo.to_csv(out_dir / "hemolysis_summary.csv", index=False)
    integrated = build_integrated_evidence(pareto, md, mic, hemo, out_dir)
    ablation_table(candidates_basic, candidates_v2, pareto, out_dir)

    make_figures(
        train,
        X_desc_train,
        embeddings,
        sequence_index,
        candidates_v2,
        pareto,
        metrics_summary,
        integrated,
        out_dir,
    )

    report = {
        "dataset": json.loads((out_dir / "dataset_audit.json").read_text(encoding="utf-8")),
        "embedding_status": embedding_status,
        "selected_model": selection,
        "output_dir": str(out_dir),
        "key_outputs": [
            "model_cv_metrics_summary.csv",
            "candidate_pareto_nsga2.csv",
            "candidate_similarity_nearest_neighbors.csv",
            "candidate_integrated_evidence.csv",
            "ablation_summary.csv",
            "figures/",
        ],
    }
    (out_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
