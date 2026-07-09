# CtxPipe baseline

This directory contains the DPPBench adapter of CtxPipe.  The adapter keeps the
core CtxPipe idea of **context-aware component-level reinforcement learning**,
while replacing only the operator space and evaluation objective with the shared
DPPBench contract.

## Alignment with original CtxPipe

| Aspect | Original CtxPipe | DPPBench adapter |
|---|---|---|
| Pipeline construction | Component/stage-level RL | Component/stage-level RL over DPPBench operator categories |
| Policies | Component-specific DQN agents | One DQN head and replay buffer per component slot |
| Context | Dataset/schema context embedding with gated plug-in | Lightweight fixed context vector plus gated context plug-in in each component Q-network |
| Actions | Component-specific transformation choices | Masked choices within the current component; optional components may `SKIP` |
| Replay | Open-Closed-Gated experience replay | OCG-style success/failure/recent replay partitions per component |
| Operator space | CtxPipe's tabular transformations | `baselines.common.utils.get_available_ops(task_type)` |
| Evaluation | Downstream model score | Shared `SearchPipelineEvaluator` / final DPPBench replay |

The adapter intentionally does **not** search arbitrary DAG parent/source edges.
DPPBench's common pipeline layer is responsible for executable DAG generation:

- `make_step(...)` creates context-aware operator parameters.
- `repair(...)` enforces mandatory operators and de-duplication.
- `truncate_optional_steps(...)` preserves the configured operator budget.
- `assign_dag_structure(...)` creates `sources`, `prev`, and `train.prev`.
- `check_pipeline(...)` validates the resulting DAG.

This avoids duplicating private common DAG helpers and fixes the previous
source/parent action semantics, where learned parent/source actions were either
invalid or overwritten during DAG assignment.

## Files

```text
baselines/CtxPipe/
├── operator_catalog.py     # thin re-export of shared common operator metadata
├── context.py              # fixed-size context encoder
├── env.py                  # component-level MDP with action masks
├── agent.py                # component-specific DQN networks and replay buffers
├── trainer.py              # masked component-level training loop
├── tester.py               # masked greedy inference
├── ctxpipe.py              # top-level runner
└── config.yaml             # default hyperparameters
```

## Notes

- `max_steps` limits the number of selected executable operators; mandatory
  operators may still be preserved by common repair when needed.
- For fairness, search-time rewards come from the shared evaluator with
  `evaluate_std_test=False`; frozen std-test replay remains the responsibility of
  the benchmark harness.
- The context encoder is lightweight to avoid the original large embedding model
  dependency, but the RL state now includes the current component and an ordered
  operator sequence so the policy can distinguish pipeline order.
