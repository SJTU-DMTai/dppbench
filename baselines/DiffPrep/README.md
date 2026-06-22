# DiffPrep Baseline

Differentiable preprocessing pipeline search inspired by *DiffPrep:
Differentiable Data Preprocessing Pipeline Search for Learning over Tabular
Data* (SIGMOD'23, [chu-data-lab/DiffPrep](https://github.com/chu-data-lab/DiffPrep)),
adapted to the dppbench operator zoo and extended to support recommendation
tasks.

## Method

DiffPrep relaxes the discrete pipeline-search problem into a continuous
optimisation over architecture parameters:

* **β matrix** (`s × m`): per-slot operator probabilities. Slot `i`'s output
  is `x_i = Σ_j β_ij · f_ij(x_{i-1})` for *soft* slots and a Gumbel-Softmax
  Straight-Through sample for *hard* slots.
* **α matrix** (`s × s`, optional Flex mode): permutation logits, Sinkhorn-
  normalised to a doubly-stochastic matrix. Disabled by default.
* **Bilevel optimisation**: outer-loop minimises validation loss w.r.t. β/α,
  inner-loop minimises training loss w.r.t. the surrogate weights `w`. We
  use a first-order DARTS approximation by default.
* **Discretisation**: argmax over each slot's β row gives the final discrete
  pipeline; structural slots' choices contribute via the ST gradient pathway.

## Differences from the paper

| Topic | Paper | This impl |
|---|---|---|
| Operator scope | 11 numeric column-wise ops (impute / normalise / outlier / discretise + identity) | All 33 operators in `dppbench/ operators/` |
| Tasks | Tabular only | Tabular + recommendation |
| Structural ops (JoinTable/DataSplit/CreateSequence/...) | n/a | Hard slots with Gumbel-Softmax + ST estimator |
| Downstream model | Differentiable (LR/MLP) trained jointly | Surrogate (LR/MLP) for search; real LightGBM/DIN for final evaluation (matches SAGA / CtxPipe / DeepPrep) |
| Order learning (Flex) | β + α, both jointly optimised | β by default; `--flex` enables α (Sinkhorn) |
| Bilevel | DARTS one-step + 2nd-order finite diff | First-order DARTS (`--eps_finite_diff` retained for forward-compat) |

## Usage

```
# Tabular
python -m baselines.DiffPrep.run_diffprep --data_name fraud_detection --n_epochs 3 --small_n 2000

# Recommendation
python -m baselines.DiffPrep.run_diffprep --data_name movielens --n_epochs 3 --small_n 2000

# Enable order learning
python -m baselines.DiffPrep.run_diffprep --data_name fraud_detection --flex
```

Outputs land in `outputs/DiffPrep/<data_name>/`:

* `best_pipeline.yaml`  -- discrete pipeline produced by argmax-projection.
* `pipeline_weights.pt` -- ``tau``, ``theta``, and surrogate state dict.
* `search_history.json` -- per-epoch losses + argmax pipeline trajectory.

## Files

| File | Role |
|---|---|
| `operator_catalog.py` | Re-exports SAGA's catalog and adds 7 missing OpSpecs. Annotates each op with `slot_kind`. |
| `slot_planner.py` | Builds canonical category-ordered slots; provides context-aware default params for the 7 DiffPrep-only ops. |
| `soft_ops.py` | Differentiable tensor implementations for the soft operator family. |
| `search_space.py` | `ContinuousPipeline` -- holds `tau` / `theta`, runs softmix or Gumbel-ST per slot. |
| `surrogate.py` | `TabularSurrogate` and `RecSurrogate` -- inner-loop learners. |
| `trainer.py` | `DiffPrepTrainer` -- bilevel optimisation loop. |
| `discretizer.py` | Argmax projection from continuous parameters to a `Pipeline`. |
| `evaluator.py` | Re-exports `CtxPipeEvaluator` so DiffPrep gets `small_n` subsampling. |
| `diffprep.py` | Top-level orchestrator. |
| `run_diffprep.py` | CLI entry. |
