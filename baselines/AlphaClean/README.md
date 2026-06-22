# AlphaClean Baseline

Best-first beam search inspired by *AlphaClean: Automatic Generation of Data
Cleaning Pipelines* (Krishnan & Wu, VLDB'19,
[sjyk/alphaclean](https://github.com/sjyk/alphaclean)), adapted to the
dppbench operator zoo and extended to support recommendation tasks.

## Method

AlphaClean treats pipeline construction as search over compositions of
*repairs* (in the paper: row-level conditional assignments). We generalise a
"repair" to a single :class:`PipelineStep` so that structural operators with
trivial predicates ("apply to all rows") fit the same abstraction. Each
iteration:

1. **Generate** -- :class:`ParameterSampler` draws a batch of operator
   instances from the catalog (an op + concrete params).
2. **Prune (learned)** -- :class:`LearnedPruner` (a Logistic Regression with
   threshold sweeping toward zero false negatives) optionally drops repairs
   that look unlikely to appear in the optimal pipeline.
3. **Expand** -- the current top-``beam_width`` pipelines are composed with
   each surviving repair to form new candidates.
4. **Evaluate** -- every new pipeline is scored by the real downstream model
   (LightGBM for tabular, DIN for rec) via :class:`CtxPipeEvaluator`.
5. **Frontier update** -- merge with the existing frontier and keep
   top-``beam_width`` distinct pipelines. ``gamma`` is retained as a
   compatibility/candidate-budget knob and is no longer the beam size.
6. **Learn** -- repairs that ended up in the best pipeline are positive
   examples; the rest are negative. Periodically refit the pruner.
7. **Early stop** if ``patience`` consecutive iterations made no improvement.

## Differences from the paper

| Topic | Paper | This impl |
|---|---|---|
| Operator scope | Data-cleaning frameworks (impute, outlier, FD repair, etc.) | All 33 dppbench operators (cleaning + featurisation + structural) |
| Tasks | Tabular cleaning | Tabular + recommendation |
| Quality function | SQL aggregations with incremental maintenance | Real downstream model AUC (matches SAGA / CtxPipe / DiffPrep) |
| Async generate-then-search | Multi-process w/ Ray | Single-process loop (the dppbench evaluator is already cached) |
| Repair format | Row-level `ca(pred, attr, v)` | A single :class:`PipelineStep` (predicate may be trivial) |
| Learned pruning | LR + threshold sweep biased toward false positives | Same |
| Parallel block search | Yes | No (the bottleneck here is downstream training, not quality eval) |

## Usage

```
# Tabular
python -m baselines.AlphaClean.run_alphaclean --data_name fraud_detection \
    --n_iters 3 --beam_width 3 --batch_per_iter 4 --small_n 1500

# Recommendation
python -m baselines.AlphaClean.run_alphaclean --data_name movielens \
    --n_iters 3 --beam_width 3 --batch_per_iter 4 --small_n 1500

# Disable learned pruner / final full-data eval
python -m baselines.AlphaClean.run_alphaclean --data_name fraud_detection --no_pruner --no_eval_full
```

Outputs land in `outputs/AlphaClean/<data_name>/`:

* `best_pipeline.yaml` -- top scoring pipeline.
* `top{1..5}_pipeline.yaml` -- frontier dump.
* `search_history.json` -- per-iter best/mean fitness, candidate counts,
  pruner stats and best ops trajectory.
* `pruner.pkl` -- pickled LR model + threshold (if sklearn available).

## Files

| File | Role |
|---|---|
| `operator_catalog.py` | Re-exports DiffPrep's catalog (33 ops). |
| `parameter_sampler.py` | Operator selection + parameter sampling. |
| `repair.py` | `Repair` dataclass and pipeline composition / featurisation. |
| `pruner.py` | Learned LR pruner with paper-style threshold sweep. |
| `searcher.py` | Best-first beam search loop. |
| `alphaclean.py` | Top-level orchestrator (`AlphaClean.run()`). |
| `run_alphaclean.py` | CLI entry. |
