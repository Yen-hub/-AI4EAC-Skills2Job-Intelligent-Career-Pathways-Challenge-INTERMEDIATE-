# Winning Solution

This folder contains the final reviewed winning pipeline.

## Entry Point

Run:

```bash
python make_oof_stacker.py
```

## Required Files

Place these challenge CSV files in this same directory before running:

- `Train.csv`
- `Test.csv`
- `Skills.csv`
- `Occupations.csv`
- `SampleSubmission.csv`

Download link:

- [Zindi challenge data page](https://zindi.africa/competitions/skills2job-intelligent-career-pathways-challenge/data)

## File Roles

- `make_oof_stacker.py`
  - final reviewed winning runner
  - builds the final stacker and writes the winning-family submissions

- `make_map_ranker_stack.py`
  - core retrieve-and-rerank engine
  - builds candidate tables and base ranker outputs

- `make_hierarchical_ranker.py`
  - hierarchy and shared utility module
  - supplies prefix and co-occurrence helpers used by the main stack

## Outputs

The script writes:

- `submission_general_recall_stacker.csv`
- `submission_general_recall_stacker_blend.csv`

## Important Note

This solution is seeded but not perfectly bitwise deterministic across all machines. Reruns are expected to stay close, but exact CSV identity may vary slightly because of GPU behavior and ranking tie ordering.
