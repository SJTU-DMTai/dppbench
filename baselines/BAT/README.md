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
- Target-column recall component:
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

The DPPBench search reward is `(target_column_recall + validation_reward) / 2`.
Each terminal candidate is evaluated using the same `SearchPipelineEvaluator`
path as SAGA, with `downstream_eval_n=10000`, a fixed seed, and fast training
enabled by default. Model parameters are fixed; no model hyperparameter search
is performed. Standard-test evaluation is disabled during search, and repeated
pipelines reuse the common evaluation cache. A failed downstream evaluation
receives zero reward. Evaluations count against the shared search budget.
This reward changes the paper's schema-only search objective; set
`use_downstream=false` to reproduce that objective. LLM-judge reward remains
optional. Final full-data evaluation is controlled separately by `eval_full`.

## Defaults

| Key | Value |
|---|---:|
| `llm_model` | `/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B` |
| `temperature` | `0.4` |
| `top_p` | `0.8` |
| `max_rollout_steps` | `10` |
| `max_depth` | `5` (official compatibility field; DPAS terminates naturally) |
| `n_action_samples` | `2` |
| `exploration_constant` | `1.0` |
| `early_stop_n_paths` | `2` |
| `reward_alpha` | `1.0` |
| `reward_beta` | `1.0` |
| `reward_gamma` | `0.0` |
| `use_downstream` | `true` |
| `downstream_eval_n` | `10000` |

`early_stop_eps` and `columns_match_threshold` remain accepted for old config
compatibility, but official-aligned inference uses exact equality.

Transformation nodes now receive sampled downstream evaluation before
expansion, so training errors and validation fitness are included in revision
prompts. Terminal scoring reuses the common evaluation cache. A training
failure permits revision even when all target columns already match.
Wide-table prompts request compact automatic column selection (`cols=None`)
and short reasoning; pipeline object lists accept `operator`, `op`, or `name`
with `params` or `kwargs`. Transformation output is limited to 1024 tokens.
The validation script defaults to 6 BAT rollouts and 2 samples per action
(`--bat-max-rollouts`, `--bat-action-samples`), under the shared 1200-second
and 48-call limits. The time limit can still stop search before six rollouts.

Schema matching and function identification use one sample each; the two
samples are reserved for transformation and revision candidate diversity.
Tabular examples use automatic missing-value and categorical-column selection.
Terminal candidates are scored before expensive revision generation, and
search reserves time based on observed LLM-call durations. If this reserve
stops further generation, evaluated terminal candidates are retained; actual
resource-budget violations and CUDA failures still propagate.

## Outputs

- `llm_trace.jsonl`: append-only prompts, raw responses, and call errors (may
  contain source samples; handle as experiment data). Written even on budget
  failure, which aborts search rather than producing empty candidates.
- Local BAT generation disables Qwen thinking; other clients retain their
  existing template defaults. JSON parsing accepts completed thinking wrappers
  and Qwen end tokens, but rejects truncated JSON/reasoning.
- `best_pipeline.yaml`: first ranked BAT candidate, with DAG metadata only.
- `best_paths.json`: up to two ranked reasoning paths.
- `agent_log.json`: reward breakdown and both ranked candidates.
- `tree.json`: complete MCTS tree with Q/N statistics.

CPU-only alignment tests:

```bash
python -m unittest baselines.BAT.test_bat_alignment -v
```

Local weight loading uses `DPPBENCH_LLM_QUANTIZATION=auto` by default: FP8
when CUDA hardware and torchao support it, otherwise unquantized weights with
a warning. Set `none` to disable quantization or `fp8` to require it strictly.
CPU loading uses FP32; this does not imply an 8B model fits available memory.
Quantization is optional; install compatible torchao/transformers on the GPU
server if required. Model choices and downstream-feedback defaults are retained.

For error metrics, `validation_reward = 1/(1+error)` (scale 1); AUC uses
its original value. `best_fitness` is the raw signed validation fitness,
while `best_search_reward` is the MCTS reward. A failed final evaluation
does not substitute an earlier sampled score.
