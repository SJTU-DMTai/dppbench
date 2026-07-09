# DiffPrep Baseline

Differentiable preprocessing pipeline search inspired by *DiffPrep: Differentiable Data Preprocessing Pipeline Search for Learning over Tabular Data* (SIGMOD'23, [chu-data-lab/DiffPrep](https://github.com/chu-data-lab/DiffPrep)). This implementation keeps the DiffPrep training structure, while adapting two benchmark-specific aspects: the operator space is DPPBench's shared operator catalog, and final quality is measured by DPPBench's common downstream evaluator/std-test protocol.

## Method

DiffPrep relaxes discrete pipeline search into continuous architecture parameters:

- **Feature-wise β / tau**: each operator-family slot has per-feature logits over candidate operators, matching the original DiffPrep design. During search, soft operators are mixed as `x_i = Σ_j β_ij · f_ij(x_{i-1})`.
- **Optional α / theta for Flex**: `flex: true` enables a Sinkhorn-normalized, feature-wise order matrix. The learned order is preserved when it satisfies DPPBench legality constraints; otherwise the executable pipeline is repaired to canonical legal order and the fallback is recorded in pipeline metadata.
- **Bilevel optimization**: architecture parameters are updated from validation loss after a virtual model-weight step on training loss. `second_order: true` enables the finite-difference Hessian-vector correction used by the original DiffPrep implementation.
- **Discretization**: feature-wise probabilities are averaged to choose one DPPBench operator per slot, then the resulting pipeline is completed and checked through `baselines.common.pipeline`.

## DPPBench adaptation boundaries

| Topic | Original DiffPrep | DPPBench implementation |
|---|---|---|
| Operator space | Tabular preprocessing functions: missing-value imputation, normalization, outlier cleaning, discretization | Shared DPPBench operator catalog via `baselines.common.utils.get_available_ops` |
| Tasks | Tabular learning with differentiable LR/MLP | Tabular + recommendation tasks with a differentiable surrogate during search |
| Search objective | Validation loss of the differentiable downstream model | Validation loss of the surrogate during search; final benchmark score from common downstream evaluator/std-test |
| Final execution | DiffPrep's own transformed tensors | DPPBench YAML `Pipeline` executed by `baselines.common.evaluator` |

## Common API usage

This baseline intentionally relies on shared APIs rather than duplicating benchmark logic:

- `get_data_ctx` for dataset context inference
- `get_available_ops` for the task-specific operator space
- `get_category_order` for public category ordering
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
| `soft_ops.py` | Differentiable tensor approximations for soft, column-wise operators. |
| `search_space.py` | Feature-wise `ContinuousPipeline` with `tau` and optional `theta`. |
| `surrogate.py` | Lightweight tabular and recommendation surrogate models for the bilevel loop. |
| `trainer.py` | DiffPrep/DARTS-style bilevel optimizer with optional second-order correction. |
| `discretizer.py` | Projects continuous choices to an executable common `Pipeline`. |
| `evaluator.py` | Thin alias for common `SearchPipelineEvaluator`. |
| `diffprep.py` | Top-level orchestrator. |
