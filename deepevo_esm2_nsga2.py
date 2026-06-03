"""
DeepEvo-Pept extension: ESM-2-ready embeddings + NSGA-II-style Pareto ranking.

This module is intentionally self-contained. If torch/transformers are installed,
it can compute real ESM-2 sequence embeddings. In lightweight environments it falls
back to deterministic sequence descriptors, while preserving the same output schema
and Pareto optimization logic.

Inputs:
  - candidates_v2.csv from the constrained evolution pipeline

Outputs:
  - candidates_esm2_nsga2.csv with Pareto front/rank and objective scores
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


HYDROPHOBIC_STRONG = set("LVIFW")
POLAR_CHARGED = set("KRDEQNSTHY")
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
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


@dataclass
class Candidate:
    source_rank: int
    source_score: float
    gen_found: int
    sequence: str
    objectives: List[float]
    pareto_rank: int = 0
    crowding_distance: float = 0.0


def read_candidates(path: Path) -> List[Candidate]:
    rows: List[Candidate] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                Candidate(
                    source_rank=int(row["Rank"]),
                    source_score=float(row["Score"]),
                    gen_found=int(row["GenFound"]),
                    sequence=row["Sequence"].strip().upper(),
                    objectives=[],
                )
            )
    return rows


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


def alternation_score(seq: str) -> float:
    """Rewards hydrophobic/polar alternation used by the constrained mutator."""
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


def sequence_descriptor_embedding(seq: str) -> List[float]:
    """Fallback embedding with stable sequence descriptors."""
    counts = Counter(seq)
    aac = [counts.get(aa, 0) / max(1, len(seq)) for aa in AA_LIST]
    return aac + [
        len(seq) / 100.0,
        net_charge(seq) / 20.0,
        hydrophobic_fraction(seq),
        hydrophobic_moment(seq),
        alternation_score(seq),
        longest_hydrophobic_run(seq) / 10.0,
    ]


def try_esm2_embeddings(seqs: Sequence[str], model_name: str) -> tuple[List[List[float]], str]:
    """Return real ESM-2 embeddings if optional dependencies are available."""
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as exc:
        return [sequence_descriptor_embedding(s) for s in seqs], f"fallback_descriptors: {exc}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    embeddings: List[List[float]] = []
    with torch.no_grad():
        for seq in seqs:
            spaced = " ".join(seq)
            batch = tokenizer(spaced, return_tensors="pt")
            out = model(**batch)
            hidden = out.last_hidden_state.squeeze(0)
            pooled = hidden.mean(dim=0).cpu().numpy().astype(float)
            embeddings.append(pooled.tolist())
    return embeddings, f"esm2:{model_name}"


def descriptor_similarity_to_seed(emb: Sequence[float], seed_emb: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(emb, seed_emb))
    na = math.sqrt(sum(a * a for a in emb))
    nb = math.sqrt(sum(b * b for b in seed_emb))
    return dot / max(1e-9, na * nb)


def objective_vector(c: Candidate, emb: Sequence[float], seed_emb: Sequence[float], seed_len: int) -> List[float]:
    seq = c.sequence
    cys_penalty = 1.0 if "C" in seq else 0.0
    run_penalty = max(0, longest_hydrophobic_run(seq) - 2) / 4.0
    hydro_excess = max(0.0, hydrophobic_fraction(seq) - 0.50)
    length_penalty = abs(len(seq) - seed_len) / max(1, seed_len)
    charge_target = 7
    charge_score = 1.0 - min(1.0, abs(net_charge(seq) - charge_target) / 12.0)
    similarity = descriptor_similarity_to_seed(emb, seed_emb)

    # NSGA-II convention here: all objectives are maximized.
    predicted_activity = c.source_score
    safety = 1.0 - min(1.0, cys_penalty + run_penalty + hydro_excess * 2.0)
    amphipathicity = 0.5 * hydrophobic_moment(seq) + 0.5 * alternation_score(seq)
    scaffold_rescue = 0.7 * similarity + 0.3 * (1.0 - length_penalty)
    synthesizability = 1.0 - min(1.0, cys_penalty + hydro_excess + run_penalty)
    return [predicted_activity, safety, amphipathicity, charge_score, scaffold_rescue, synthesizability]


def dominates(a: Candidate, b: Candidate) -> bool:
    better_or_equal = all(x >= y for x, y in zip(a.objectives, b.objectives))
    strictly_better = any(x > y for x, y in zip(a.objectives, b.objectives))
    return better_or_equal and strictly_better


def non_dominated_sort(candidates: List[Candidate]) -> List[List[Candidate]]:
    fronts: List[List[Candidate]] = []
    domination_counts = {id(c): 0 for c in candidates}
    dominated = {id(c): [] for c in candidates}
    first_front: List[Candidate] = []

    for p in candidates:
        for q in candidates:
            if p is q:
                continue
            if dominates(p, q):
                dominated[id(p)].append(q)
            elif dominates(q, p):
                domination_counts[id(p)] += 1
        if domination_counts[id(p)] == 0:
            p.pareto_rank = 1
            first_front.append(p)

    fronts.append(first_front)
    rank = 1
    while fronts[-1]:
        next_front: List[Candidate] = []
        for p in fronts[-1]:
            for q in dominated[id(p)]:
                domination_counts[id(q)] -= 1
                if domination_counts[id(q)] == 0:
                    q.pareto_rank = rank + 1
                    next_front.append(q)
        rank += 1
        fronts.append(next_front)
    return fronts[:-1]


def assign_crowding_distance(front: List[Candidate]) -> None:
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


def write_output(path: Path, candidates: Iterable[Candidate], embedding_status: str) -> None:
    fields = [
        "ParetoRank",
        "CrowdingDistance",
        "SourceRank",
        "SourceScore",
        "GenFound",
        "Sequence",
        "ActivityObjective",
        "SafetyObjective",
        "AmphipathicityObjective",
        "ChargeObjective",
        "ScaffoldRescueObjective",
        "SynthesizabilityObjective",
        "EmbeddingStatus",
    ]
    ordered = sorted(candidates, key=lambda c: (c.pareto_rank, -c.crowding_distance, c.source_rank))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in ordered:
            writer.writerow(
                {
                    "ParetoRank": c.pareto_rank,
                    "CrowdingDistance": c.crowding_distance,
                    "SourceRank": c.source_rank,
                    "SourceScore": c.source_score,
                    "GenFound": c.gen_found,
                    "Sequence": c.sequence,
                    "ActivityObjective": c.objectives[0],
                    "SafetyObjective": c.objectives[1],
                    "AmphipathicityObjective": c.objectives[2],
                    "ChargeObjective": c.objectives[3],
                    "ScaffoldRescueObjective": c.objectives[4],
                    "SynthesizabilityObjective": c.objectives[5],
                    "EmbeddingStatus": embedding_status,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="candidates_v2.csv")
    parser.add_argument("--output", default="candidates_esm2_nsga2.csv")
    parser.add_argument("--seed", default="KAWNLRGSAREKAIKNEKLYIFATSGKLAALKPK")
    parser.add_argument("--esm2-model", default="facebook/esm2_t6_8M_UR50D")
    args = parser.parse_args()

    candidates = read_candidates(Path(args.input))
    seqs = [c.sequence for c in candidates]
    embeddings, status = try_esm2_embeddings(seqs + [args.seed], args.esm2_model)
    seed_emb = embeddings[-1]

    for c, emb in zip(candidates, embeddings[:-1]):
        c.objectives = objective_vector(c, emb, seed_emb, len(args.seed))

    fronts = non_dominated_sort(candidates)
    for front in fronts:
        assign_crowding_distance(front)

    write_output(Path(args.output), candidates, status)
    print(f"embedding_status={status}")
    print(f"wrote={args.output}")
    print("top Pareto candidates:")
    for c in sorted(candidates, key=lambda x: (x.pareto_rank, -x.crowding_distance, x.source_rank))[:10]:
        print(c.pareto_rank, c.source_rank, f"{c.source_score:.3f}", c.sequence)


if __name__ == "__main__":
    main()
