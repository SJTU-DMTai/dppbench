# DataMaster Baseline

LLM-powered, tree-based agentic data preparation for dppbench, ported from
**DataMaster: Data-Centric Autonomous AI Research**
([paper](../../papers/DataMaster%20Data-Centric%20Autonomous%20AI%20Research.pdf),
[code](https://github.com/sjtu-sai-agents/DataMaster)).

The DataMaster paper organises agentic search around a **DataTree** with
two kinds of nodes: red nodes that explore *new* data sources, and black
nodes that *refine* the dataset by running preprocessing scripts. A
**Global Memory** records `(D_v, y_v, φ_v)` for every node and a
**UCB-style scheduler** picks the next node to expand.

## Scope adjustments

dppbench fixes the input data per task and forbids external data
collection, so this baseline implements DataMaster with two scope
adjustments:

| Paper / source-code mechanism | Here |
|---|---|
| DataTree (multi-branch search) | ✅ kept |
| Black nodes (data refinement) | ✅ kept |
| **Red nodes (external data exploration)** | ❌ removed |
| **Data Pool** | ❌ removed |
| Global Memory + cross-branch retrieval | ✅ kept (parent + siblings + global top-K) |
| UCB scheduling with decaying `c_t` | ✅ kept (linear / exponential / piecewise) |
| Reward backpropagation | ✅ kept |
| **LLM writes Python code** | ❌ replaced with **operator selection** |
| Operator coverage | ✅ all **77** ops from `dppbench/ operators/` |

The LLM never writes Python. For every black node it emits a single
`<solution>Op1(arg=value) --> Op2(...) --> Terminate</solution>` chain
drawn from the SAGA 77-op catalog. The chain is parsed via
`baselines.DeepPrep.tree_agent.chain_to_steps` and then serialised back to
the same prev-only DAG YAML used by every other baseline
(`dag.sources`, `dag.ops[*].prev`, `dag.train.prev`), so the result is
directly consumable by `data.run_pre_process(yaml_path)`.

## Files

```
baselines/DataMaster/
├── __init__.py             # exposes DataMaster
├── operator_catalog.py     # re-export SAGA 77-op catalog
├── data_tree.py            # NodeRecord / DataTree / backpropagate
├── memory.py               # GlobalMemory.retrieve / format_context
├── scheduler.py            # UCBScheduler + c_t decay
├── prompts.py              # SYSTEM_DATAMASTER + user prompt
├── agent.py                # DataMasterAgent main loop
├── data_master.py          # top-level DataMaster class
├── evaluator.py            # DataMasterEvaluator (CtxPipeEvaluator subclass)
└── run_data_master.py      # CLI entry point
```

## Quick start

```bash
# Tabular smoke test (API credentials are loaded from project apikeys.json)
python -m baselines.DataMaster.run_data_master \
    --data_name fraud_detection --llm_model gpt-4o-mini \
    --max_iterations 3 --k_black 2 --max_chain_len 4 \
    --small_n 1000 --downstream_eval_n 500 --no_eval_full

# Recommendation smoke test
python -m baselines.DataMaster.run_data_master \
    --data_name movielens --llm_model gpt-4o-mini \
    --max_iterations 3 --k_black 2 --max_chain_len 4 \
    --small_n 1000 --downstream_eval_n 500 --no_eval_full
```

The runner emits four artefacts under `outputs/DataMaster/<data_name>/`:

* `best_pipeline.yaml` — final pipeline in prev-only DAG YAML format.
* `agent_log.json` — full LLM transcript, solution attempts, scheduler config.
* `tree.json` — every node in the DataTree (UCB stats, fitness, error, findings).
* `memory.json` — GlobalMemory log.

## Key hyper-parameters

| flag | meaning | default |
|---|---|---|
| `--max_iterations` | outer UCB select-and-expand loop | 5 |
| `--k_black` | black children spawned per parent per iter | 3 |
| `--max_chain_len` | max ops in one delta chain | 6 |
| `--max_depth` | max depth of any branch | 6 |
| `--max_solution_attempts` | per-child LLM retries on rejection | 2 |
| `--c_initial`, `--decay` | UCB exploration coefficient and its schedule | 1.414, linear |
| `--reward_kind` | `fitness` (paper Eq.2) or `improvement` (±1 vs parent) | fitness |
| `--memory_top_k` | global top-K best nodes injected into prompt | 3 |
| `--small_n` / `--downstream_eval_n` | sandbox / agent-eval subsample sizes | 0 / 3000 |
