# Learn2Clean Baseline

Tabular Q-learning + fixed epsilon-greedy exploration as proposed in *Learn2Clean:
Optimizing the Sequence of Tasks for Web Data Preparation* (Berti-Équille,
Information Systems / WWW 2019,
[LaureBerti/Learn2Clean](https://github.com/LaureBerti/Learn2Clean)),
adapted to the dppbench operator zoo and extended to recommendation tasks.

## Method

Learn2Clean models pipeline construction as a Markov Decision Process:

* **State**: the set of operators already applied, plus the index of the last
  operator. We use `(frozenset(applied_op_idx), last_op_idx)` so the same set
  of ops reached via different orders shares its Q-row when meaningful.
* **Action**: pick the next operator from the dppbench catalog (33 ops) or
  STOP.
* **Reward**: per-step downstream-AUC delta, clipped by an `improvement_eps`
  band into `{+r_max, 0, -r_max}`; an illegal/duplicated/no-default action
  yields `illegal_reward`. Terminal states earn an additional
  `reward_max * (final_auc - baseline_auc)`.

Each episode runs the loop:

1. **Reset** -- empty pipeline, baseline AUC measured once via `repair()`.
2. **Select action** -- fixed `epsilon=0.05` epsilon-greedy: choose a random
   action with probability 0.05, otherwise choose `argmax Q(s, ·)`.
3. **Step** -- legality check (duplicate / no_default_params / illegal_order),
   apply via `PipelineStep`, evaluate, roll back on NaN.
4. **Update** -- tabular Q-learning:
   `Q(s,a) ← Q(s,a) + lr * (r + γ·max Q(s',a') - Q(s,a))`.

After all episodes the best (and top-K) pipelines are repaired and persisted.

## Differences from the paper

| Topic | Paper | This impl |
|---|---|---|
| Operator scope | 5 cleaning families (impute, outlier, dedup, FD, normalize) | All 33 dppbench operators (cleaning + featurisation + structural + rec/seq) |
| Tasks | Tabular cleaning | Tabular + recommendation |
| Quality function | Per-task normalised metrics per family | Real downstream AUC delta (matches SAGA / CtxPipe / DiffPrep / AlphaClean) |
| Exploration | Fixed ε-greedy (`epsilon=0.05`) | Same |
| Algorithm | Tabular Q-learning | Same (dict-based Q-table) |
| State | Set of applied ops | `(frozenset(ops), last_op_idx)` (sharable across reorderings) |
| Reward | Task-aware ±r_max | Per-step ±r_max (AUC delta) + terminal `r_max·(final-base)` |
| Loop | Multiple episodes | Same (default 12 episodes) |

## Usage

```bash
# Tabular
python scripts/evaluate_baseline.py --baseline Learn2Clean --data_names fraud_detection

# Recommendation
python scripts/evaluate_baseline.py --baseline Learn2Clean --data_names movielens
```

Outputs land in `outputs/Learn2Clean/<data_name>/`:

* `best_pipeline.yaml` -- top scoring pipeline.
* `top{1..5}_pipeline.yaml` -- top-K dump.
* `train_history.json` -- per-episode reward / fitness / ops / epsilon.
* `q_table.json` -- learned Q values keyed by encoded state.

## Files

| File | Role |
|---|---|
| `operator_catalog.py` | Re-exports the shared common operator catalog from `baselines.common.op_spec`. |
| `env.py` | `Learn2CleanEnv` -- MDP wrapper; legal/duplicate/illegal-order checks, eval-and-rollback, AUC-delta reward. |
| `agent.py` | `TabularQAgent` -- dict-based Q-table, fixed epsilon-greedy policy, JSON save/load. |
| `trainer.py` | `Learn2CleanTrainer` -- episode loop and best/top-K bookkeeping. |
| `learn2clean.py` | Top-level orchestrator (`Learn2Clean.run()`); wires evaluator/ctx/env/agent/trainer and writes artefacts. |
| `__init__.py` | Re-exports `Learn2Clean`. |
| `README.md` | This document. |
