# CtxPipe Baseline

A reimplementation of **CtxPipe: Context-aware Data Preparation Pipeline
Construction for Machine Learning** (SIGMOD 2025) inside `dppbench`.

Reference: <https://github.com/ctxpipe/ctxpipe>

## Method

CtxPipe formulates pipeline construction as a Markov Decision Process and
trains a **Deep Q-Network (DQN)** to select operators step by step. Compared
to genetic-search baselines (e.g. SAGA), CtxPipe **learns** a policy that
adapts the next operator to the current dataset's *context* (schema /
statistics / task type).

| Component     | Original paper                              | This implementation                                    |
|---------------|----------------------------------------------|--------------------------------------------------------|
| Operator pool | 11 cleaning operators                        | All 27 operators in `dppbench/ operators/` (cleaning + encoding + feature-gen + sequence + split + sampling + ...) |
| Tasks         | Tabular only                                 | **Tabular (LightGBM/AUC) and Recommendation (DIN/AUC)** |
| Context plug-in | GTE-large embedding (~600 MB)              | Lightweight 32-dim schema/statistics encoder            |
| Replay buffer | OCG (Open-Closed-Gated) experience replay    | Vanilla replay buffer                                   |
| Optimisation  | DQN + target network + ε-greedy              | DQN + target network + ε-greedy (same)                  |
| Reward        | Downstream metric at episode end             | LightGBM/DIN AUC at episode end (same)                  |

The two simplifications (lightweight context encoder, vanilla replay) keep the
core RL training loop intact and avoid heavyweight dependencies. They are
clearly marked in the code.

## Files

```
baselines/CtxPipe/
├── operator_catalog.py     # re-exports SAGA's catalog (27 operators)
├── context.py              # 32-dim schema-feature encoder
├── env.py                  # PipelineEnv: state / action / reward
├── agent.py                # QNetwork + ReplayBuffer + DQNAgent (PyTorch)
├── trainer.py              # RL training loop with ε-decay + target update
├── tester.py               # Greedy inference of the learned policy
├── evaluator.py            # PipelineEvaluator with optional small_n subsampling
├── ctxpipe.py              # Top-level CtxPipe class
└── run_ctxpipe.py          # CLI entry point
```

## Usage

```bash
# Tabular smoke test (fast)
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python -m baselines.CtxPipe.run_ctxpipe \
    --data_name fraud_detection --n_episodes 3 --max_steps 5 --small_n 3000

# Recommendation smoke test (fast)
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python -m baselines.CtxPipe.run_ctxpipe \
    --data_name movielens --n_episodes 3 --max_steps 5 --small_n 3000

# Full run (default hyperparameters)
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python -m baselines.CtxPipe.run_ctxpipe \
    --data_name home_credit --n_episodes 20 --max_steps 8 --small_n 5000
```

CLI arguments:

| Flag             | Default | Description                                                          |
|------------------|---------|----------------------------------------------------------------------|
| `--data_name`    | `movielens` | One of the six dppbench datasets                                |
| `--n_episodes`   | `20`    | Number of RL episodes                                                |
| `--max_steps`    | `8`     | Maximum operators per pipeline                                       |
| `--small_n`      | `5000`  | Subsample size during RL training (`0` = full data)                  |
| `--eval_full`    | True    | Re-evaluate the final pipeline on the full dataset                   |
| `--seed`         | `42`    | Reproducibility seed                                                  |
| `--output_dir`   | auto    | Where `best_pipeline.yaml` and `q_network.pt` are saved              |

The trainer always saves the final pipeline to
`outputs/CtxPipe/<data_name>/best_pipeline.yaml` and the Q-network weights
alongside.
