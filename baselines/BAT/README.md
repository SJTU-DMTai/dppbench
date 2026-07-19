# BAT Baseline

This adapter follows the inference-time algorithm in
`ZJU-DAILY/BAT` while retaining the DPPBench benchmark contract.

## Preserved BAT behavior

- DPAS node types: Root, SchemaMatch, IdentifyColumnFunctions,
  Transformation, TransformationRevision, and End.
- The official node-to-action transition table and no-repeat action rule.
- UCB selection `Q/N + c * sqrt(log(parent.N)/child.N)` with the first
  unvisited child selected before UCB.
- Expansion of every legal action followed by shuffled child order.
- Uniform random simulation until a natural End node.
- One terminal EPO reward propagated through all ancestors.
- Exact `reward == 1.0` early stopping after two perfect paths.
- Up to two returned candidates; missing perfect candidates are filled by
  terminal average value `Q/N`. DPPBench evaluates the first ranked candidate.
- Official target-column recall reward:
  `|result_columns intersect target_columns| / |target_columns|`.
- Exact set equality for the `columns_match` branch decision.
- Fenced-JSON prompt/response shape and a second LLM call to repair malformed
  transformation JSON.

## Required DPPBench adaptations

Official BAT generates arbitrary pandas statements. DPPBench instead requires
the LLM to place a shared operator chain in the JSON `pipeline` field. The
candidate is executed through the shared sandbox and serialized as common DAG
YAML. Illegal candidates are rejected; operators are never silently inserted,
reordered, or repaired.

Official BAT receives an explicit target schema. DPPBench derives the target
schema from `DataContext` and the downstream model-input contract because tasks
do not currently expose a separate target-schema artifact. No target table
instances or frozen standard-test rows are exposed to the search.

The downstream LightGBM/DIN metric is computed after search for benchmark
reporting. It is not part of MCTS reward by default. The optional downstream
and LLM-judge reward channels remain available only for explicitly declared
ablations.

## Defaults

| Key | Value |
|---|---:|
| `llm_model` | `qwen2.5-coder-32b-instruct` |
| `temperature` | `0.1` |
| `top_p` | `0.8` |
| `max_rollout_steps` | `10` |
| `max_depth` | `5` (official compatibility field; DPAS terminates naturally) |
| `n_action_samples` | `1` |
| `exploration_constant` | `1.0` |
| `early_stop_n_paths` | `2` |
| `reward_alpha` | `1.0` |
| `reward_beta` | `0.0` |
| `reward_gamma` | `0.0` |
| `use_downstream` | `false` |

`early_stop_eps` and `columns_match_threshold` remain accepted for old config
compatibility, but official-aligned inference uses exact equality.

## Outputs

- `best_pipeline.yaml`: first ranked BAT candidate, with DAG metadata only.
- `best_paths.json`: up to two ranked reasoning paths.
- `agent_log.json`: reward breakdown and both ranked candidates.
- `tree.json`: complete MCTS tree with Q/N statistics.

CPU-only alignment tests:

```bash
python -m unittest baselines.BAT.test_bat_alignment -v
```
