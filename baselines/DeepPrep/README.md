# DeepPrep Baseline

This adapter runs a released DeepPrep checkpoint with execution-grounded tree
inference over DPPBench's shared operator space. Model training is intentionally
out of scope; `rl_trainer.py` remains a non-functional placeholder.

## Alignment

| Aspect | Implementation |
|---|---|
| Model action | `<plan>` followed by exactly one `<expand>` or `<answer>` |
| Environment action | `<execute>` is generated only by the sandbox |
| Tree state | Materialized table-set snapshot per node |
| Tree edge | One successfully executed operator instance |
| Parent selection | Full root-to-node path including parameters |
| Backtracking | Non-local parent selection in `<expand>` |
| Termination | Budget-limited candidate comparison in AUC mode; validated answer in schema-only mode |
| Search feedback | Schema, execution/training errors, and sampled validation AUC |
| Downstream metric | Fixed-parameter training on up to 10,000 sampled rows; no HPO or standard-test scoring during search |
| Finalization | DAG metadata assignment and hard validation; no semantic repair |

The default AUC-feedback mode is a DPPBench adaptation, not schema-only paper
inference. It reuses the shared search evaluator, caches candidate scores, and
selects the highest validation AUC within the existing search/evaluation budget.
LLM observation sampling (`small_n`) is independent of `downstream_eval_n`.
Set `downstream_feedback: false` to restore schema-only search. Frozen standard
test replay and final reporting remain separate from the search objective.

DPPBench necessarily differs from the original method in its shared operator
space and downstream benchmark objective. `TargetSchema` is derived from the
public `DataContext` and model-input contract; the target task's `prepare.yaml`
is never exposed to the agent.

## Checkpoint

Set the released checkpoint either in `config.yaml` or through:

```bash
export DEEPPREP_MODEL=/absolute/path/to/the/released/checkpoint
```

The local loader requires the checkpoint's own tokenizer and `chat_template`.
It preserves special action tokens during decoding. A base Qwen model or a
generic OpenAI API model is not treated as an equivalent replacement.

Important defaults:

| Key | Default |
|---|---|
| `llm_backend` | `local` |
| `temperature` | `0.01` |
| `top_p` | `1.0` |
| `max_explore_turn` | `5` |
| `downstream_feedback` | `false` (deprecated compatibility key) |

## Outputs

`best_pipeline.yaml` is the exact agent-selected path with only deterministic
DAG IDs/edges attached. `agent_log.json` records checkpoint and generation
settings plus the complete assistant/environment transcript. `tree.json`
stores parameterized paths and all materialized tree nodes.

## Verification

The protocol tests are CPU-only and require no dataset:

```bash
python -m unittest baselines.DeepPrep.test_deepprep_protocol -v
```
