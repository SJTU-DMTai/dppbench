# HAIPipe baseline for DPPBench

This adapter ports HAIPipe's core method structure into DPPBench without using
its notebook code-insertion runtime.

## Method mapping

| HAIPipe concept | DPPBench implementation |
|---|---|
| HI-pipeline | Common mandatory scaffold from an empty `Pipeline` |
| AI-pipeline | Component-level DQN over shared DPPBench operators |
| HAI-pipeline | Enumerated insertions of AI ops into the HI scaffold |
| Candidate selection | Shared `SearchPipelineEvaluator` downstream fitness |
| Legality | `finalize_pipeline(...)` and `check_pipeline(...)` |

The standard benchmark mode uses `hi_source: mandatory` so HAIPipe receives no
extra notebook or task `prepare.yaml` prior.  This preserves the DPPBench rule
that every baseline starts from the same task information.

## Common API usage

The implementation relies on shared APIs for all executable details:

- `get_data_ctx(...)`
- `get_available_ops(...)`
- `get_category_order(...)`
- `make_step(...)`
- `finalize_pipeline(...)`
- `check_pipeline(...)`
- `SearchPipelineEvaluator(...)`

It does not maintain a private operator catalog or private parameter factory.

## Files

- `haipipe.py`: top-level runner and output contract
- `hi_pipeline.py`: HI scaffold construction
- `operator_mapping.py`: HAIPipe components to DPPBench categories
- `env.py`: component-level AI-pipeline environment
- `trainer.py`: DQN training and greedy inference
- `combiner.py`: HI + AI enumeration/sampling and best candidate selection
