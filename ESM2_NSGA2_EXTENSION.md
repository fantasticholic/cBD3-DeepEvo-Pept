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
python3 deepevo_esm2_nsga2.py --input candidates_v2.csv --output candidates_esm2_nsga2.csv
```

Current embedding status:

```text
fallback_descriptors: No module named 'torch'
```

This means the Pareto ranking has been executed locally, but the current machine
did not have the optional deep learning stack needed for real ESM-2 inference.
After installing `torch` and `transformers`, the same script will automatically
use:

```text
facebook/esm2_t6_8M_UR50D
```

## Manuscript wording boundary

Accurate wording:

> We extended DeepEvo-Pept with an ESM-2-ready embedding module and an NSGA-II-style Pareto ranking layer.

Use this only after real ESM-2 has been executed:

> ESM-2 embeddings were computed for all candidate peptides.

