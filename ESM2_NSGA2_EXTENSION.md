# DeepEvo-Pept ESM-2 / NSGA-II Extension

This extension adds a reproducible multi-objective ranking layer to the existing
constrained evolution pipeline.

## Implemented in this workspace

- `deepevo_esm2_nsga2.py`
  - Reads `candidates_v2.csv`.
  - Computes ESM-2-ready sequence embeddings.
  - Uses real ESM-2 embeddings when `torch` and `transformers` are installed.
  - Falls back to deterministic sequence descriptors when the deep learning
    stack is unavailable.
  - Scores six objectives:
    1. predicted activity from the constrained random forest pipeline
    2. safety penalty against Cys, long hydrophobic runs, and excessive hydrophobicity
    3. amphipathicity via hydrophobic moment and hydrophobic/polar alternation
    4. charge objective around a cationic AMP target
    5. scaffold rescue similarity to cBD3-ABU
    6. synthesizability
  - Applies NSGA-II-style non-dominated sorting and crowding distance ranking.
  - Writes `candidates_esm2_nsga2.csv`.

## Current local run

Command:

```bash
python deep_evo_top_journal_pipeline.py
```

Current embedding status:

```text
cache:facebook/esm2_t6_8M_UR50D
```

This means real ESM-2 embeddings were computed previously on this workspace and
the reproducible pipeline reused the cache for the current run. The full
top-journal evidence generator writes:

- `results/top_journal/model_cv_metrics_summary.csv`
- `results/top_journal/candidate_pareto_nsga2.csv`
- `results/top_journal/candidate_integrated_evidence.csv`
- `results/top_journal/ablation_summary.csv`
- `results/top_journal/figures/`

Validation command:

```bash
python validate_top_journal_outputs.py
```

## Manuscript wording boundary

Accurate wording:

> We extended DeepEvo-Pept with real ESM-2 peptide embeddings, descriptor/ESM-2/hybrid model ablation, and an NSGA-II-style multi-objective ranking layer.

The current run supports:

> ESM-2 embeddings were computed for all candidate peptides.
