# SAGA Baseline

A Python re-implementation of the core ideas of:

> Siddiqi, S., Kern, R., & Boehm, M. (2024). **SAGA: A Scalable Framework for Optimizing Data Cleaning Pipelines for Machine Learning Applications**. SIGMOD 2024.

The original SAGA is implemented on top of Apache SystemDS (DML). This module
ports the *method* (not the runtime) into the dppbench codebase so it can drive
the existing operator library and downstream models.

## Method overview

SAGA performs a two-level pipeline search:

1. **Logical pipeline enumeration** — a genetic algorithm searches for the best
   sequence of preprocessing operators (their type and order). Each candidate
   is evaluated by training the downstream model and reading its validation
   metric (AUC).
2. **Physical pipeline tuning** — for the top-K logical pipelines from
   step (1), the hyperparameters of each operator are tuned via random search.

Pipelines are *downstream-model aware*: the fitness signal comes directly from
training a LightGBM (tabular tasks) or DIN (recommendation tasks) model and
evaluating on the validation/test split.

## What is included / simplified

| Aspect | Original SAGA | This implementation |
|--------|---------------|---------------------|
| Logical search | Genetic algorithm | Genetic algorithm (same) |
| Physical search | Hyperband | Random search (lighter, no extra deps) |
| Pruning | Lineage / monotonicity-aware | Fitness caching + early stopping |
| Runtime | SystemDS DML, distributed | Python, single-machine |
| Operator coverage | Cleaning operators only | **All 28 operators in `dppbench/ operators/` (cleaning + feature eng + rec ops)** |
| Tasks | Tabular ML | Tabular **and** Recommendation (DIN) |

## Usage

```bash
# Tabular task
python -m baselines.SAGA.run_saga \
    --data_name fraud_detection \
    --population_size 8 \
    --n_generations 3 \
    --top_k 3 \
    --n_physical_trials 3

# Recommendation task
python -m baselines.SAGA.run_saga \
    --data_name movielens \
    --population_size 8 \
    --n_generations 3 \
    --top_k 3 \
    --n_physical_trials 3
```

Supported datasets: `home_credit`, `fraud_detection`, `amazon_beauty`,
`movielens`, `yelp`, `tenrec`.

The best pipeline (YAML) is written to
`outputs/SAGA/<dataset>/best_pipeline.yaml`.
