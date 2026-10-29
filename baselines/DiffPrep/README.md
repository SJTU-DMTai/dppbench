# DiffPrep Baseline

Differentiable preprocessing pipeline search inspired by *DiffPrep: Differentiable Data Preprocessing Pipeline Search for Learning over Tabular Data* (SIGMOD'23, [chu-data-lab/DiffPrep](https://github.com/chu-data-lab/DiffPrep)). This implementation keeps the DiffPrep training structure, while adapting two benchmark-specific aspects: the operator space is DPPBench's shared operator catalog, and final quality is measured by DPPBench's common downstream evaluator/std-test protocol.

## Method

### Explicit relational adapter (optional)

The supplied `baselines/DiffPrep/config.yaml` now enables this Berka experiment:

```yaml
join_overrides:
  berka:
    table: order
    key_col: account_id
    method: agg
    prefix: null
    max_cols: null
```

This is an explicit experiment configuration, not a default recommendation for
every dataset. Set `join_overrides: {}` to restore all legacy seeded choices.
For account use `table: account, method: key`. `table`, `key_col`
and `method` are required; `prefix` and `max_cols` are optional. Only left joins
are supported. The selected table must have a direct join key in the context;
key joins require a unique right key. Invalid overrides fail rather than silently
falling back to a random table. Overrides currently support tabular tasks only.

Unlisted datasets retain the previous independent `Random(seed)` adapter draw
and candidate ordering. Old custom config files without the new key still work.
Python callers can supply the same mapping as `DiffPrep(join_overrides=...)`.
The candidate operators, gradient optimizer, split, sampling and model settings
are unchanged. The existing evaluation command reads this configuration without
additional CLI flags. Use a fresh output directory/search; reused old YAMLs do
not apply new adapter configuration.

The adapter executes before search and is copied into the exported pipeline.
Before saving, checks compare its parameters and the serialized YAML source
wiring with the search adapter. `search_contract.json` retains version 3 and adds
`join_selection` (mode, seed, candidate order, actual parameters, partition row
counts and added columns) and `join_replay_checked: true`. This asserts join
replay only, not equivalence of all soft operators and their real counterparts.

Sync `data_protocol.py`, `diffprep.py`, and `config.yaml` for runtime changes;
also sync this README and `test_explicit_join.py` for documentation and tests.

DiffPrep relaxes discrete pipeline search into continuous architecture parameters:

- **Feature-wise β / tau**: each operator-family slot has per-feature logits over candidate operators, matching the original DiffPrep design. During search, soft operators are mixed as `x_i = Σ_j β_ij · f_ij(x_{i-1})`.
- **Optional α / theta for Flex**: `flex: true` enables a Sinkhorn-normalized, feature-wise order matrix. Mandatory preparation stages stay fixed, matching the original treatment of missing-value handling; shared operator-level and DAG legality checks still apply.
- **Bilevel optimization**: architecture parameters are updated from validation loss after a virtual model-weight step on training loss. The finite-difference Hessian-vector correction is enabled by default, and the best-validation checkpoint is restored before discretization.
- **Discretization**: feature-wise argmax choices are materialized per column when the shared operator accepts `cols`. Dataframe-wide operators use a deterministic majority projection and retain all feature-wise choices in pipeline metadata.

## DPPBench adaptation boundaries

| Topic | Original DiffPrep | DPPBench implementation |
|---|---|---|
| Operator space | Tabular preprocessing functions: missing-value imputation, normalization, outlier cleaning, discretization | Supported tensor operators from the shared catalog; fixed structural adapters at export |
| Tasks | Tabular classification with differentiable LR | Tabular classification with multinomial LR + recommendation tasks with a differentiable recommendation surrogate |
| Search objective | Validation loss of the differentiable downstream model | Validation loss of the surrogate during search; final benchmark score from common downstream evaluator/std-test |
| Final execution | DiffPrep's own transformed tensors | DPPBench YAML `Pipeline` executed by `baselines.common.evaluator` |

## Common API usage

This baseline intentionally relies on shared APIs rather than duplicating benchmark logic:

- `get_data_ctx` for dataset context inference
- `get_available_ops` for the task-specific operator space
- `get_category_order` for public category grouping (without order constraints)
- `make_step` for context-aware, constructor-compatible operator parameters
- `repair`, `assign_dag_structure`, and `check_pipeline` for executable pipeline construction
- `SearchPipelineEvaluator` for final downstream evaluation and `small_n` support

## Usage

```bash
# Tabular
python scripts/evaluate_baseline.py --baseline DiffPrep --data_names fraud_detection

# Recommendation
python scripts/evaluate_baseline.py --baseline DiffPrep --data_names amazon_beauty --model DIN
```

Outputs are written under the selected evaluation output directory, e.g. `outputs/eval_std_test/DiffPrep/<task>_<model>/`:

- `best_pipeline.yaml` -- discrete DPPBench pipeline from argmax projection
- `pipeline_weights.pt` -- `tau`, optional `theta`, `alpha`, and surrogate weights
- `search_history.json` -- per-epoch train/validation losses and argmax pipeline trajectory

## Files

| File | Role |
|---|---|
| `operator_catalog.py` | DiffPrep metadata adapter over the shared operator catalog; annotates slot kind. |
| `slot_planner.py` | Builds category slots from common operator-space/order APIs and common `make_step`. |
| `soft_ops.py` | Shared-operator tensor adapters used with the original centered numerical-difference straight-through gradient. |
| `search_space.py` | Feature-wise `ContinuousPipeline` with `tau` and optional `theta`. |
| `surrogate.py` | Lightweight tabular and recommendation surrogate models for the bilevel loop. |
| `trainer.py` | DiffPrep/DARTS-style bilevel optimizer with the official second-order correction enabled by default. |
| `discretizer.py` | Projects continuous choices to an executable common `Pipeline`. |
| `evaluator.py` | Thin alias for common `SearchPipelineEvaluator`. |
| `diffprep.py` | Top-level orchestrator. |

## Search protocol audit (2026-10-12)

This remains a DPPBench adaptation, not an exact execution-equivalent reproduction
of all shared operators. The tabular surrogate is a single linear head; final
LightGBM is a different optimization target. Soft forwards are approximations
and their selected YAML operators are refitted by the shared evaluator.

Unsupported optional structural operations are no longer assigned artificial
candidate-index-dependent gates. Mixed categories retain their supported soft
choices. Required hard adapters have no trainable choice and act as identity
in tensor search; their real execution happens only in the exported dataframe
pipeline. A JoinTable inserted by structural repair is therefore not a
learned selection based on the quality of joined history features.

DiffPrep now passes its seeded RNG through structural repair (including joins
and mandatory tails). Other callers retain their existing behavior unless they
supply the new optional rng argument. The saved `search_contract.json` and returned `search_contract`
record version 2, actual search feature names, exported joins and the structural
adapter policy. Inspect the join's aux_df/method/key_col before interpreting a
low Berka score; it is not necessarily an account-only pipeline.

Tabular search loads only train_frozen.parquet, validates split markers and uses
the task's canonical train/validation split. It never opens std_test.parquet.
small_n limits the training partition after splitting, not the validation set.
A previous load_std_test_frozen call loaded held-out labels even though they
were subsequently excluded; this was unnecessary access, not evidence that
those labels contributed to the training loss.

These corrections change DiffPrep search/export behavior on all supported tasks.
Rerun its experiments rather than comparing old and new results as one version.
No dataset-specific Berka history features were added to DiffPrep.
