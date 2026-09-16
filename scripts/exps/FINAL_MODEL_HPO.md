# Final model hyperparameter search

All implemented experiments using common.sh/run_eval enable --final_hpo.
Baseline pipeline search finishes before tuning starts. Baseline search never
invokes this tuner.

The fixed common_hparams_v2 space uses nine full-training trials per
baseline. The nine configurations are curated deterministic candidates rather
than a full Cartesian product, so final evaluation cost remains bounded.

- LightGBM tunes learning rate, estimator count, leaf count, maximum depth,
  minimum child samples, row/column sampling and L1/L2 regularization.
- Recommendation models tune learning rate, embedding dimension, DNN hidden
  units, dropout, embedding regularization, weight decay and batch size.
- Graph models tune learning rate, hidden dimension, layer count, dropout and
  weight decay.
- LSTM/GRU tune learning rate, hidden size, layer count, dropout and batch size.
- Transformer tunes learning rate, model dimension, attention heads, layer
  count, feed-forward dimension, dropout and batch size.
- MLP tunes learning rate, hidden units, dropout, weight decay and batch size.
- TabTransformer, FTTransformer and SAINT tune learning rate, model dimension,
  attention heads, layer count, feed-forward dimension, dropout, weight decay
  and batch size.

Other model parameters and training budgets retain the task defaults. Thus each
task/model pair has identical candidate configurations across baselines.

Trials evaluate the fixed pipeline using validation only. The first metric in
train.metrics determines selection; error metrics are minimized, other metrics
maximized. Failed or non-finite trials cannot win. Ties select the first trial.
A fresh model is trained with the winning parameters and the same script seed,
then scored on the frozen standard test. Standard-test results never select
hyperparameters. The existing train/validation and early-stopping protocol is
retained for this final training.

Each baseline/task directory contains final_hpo.json (parameters, validation
metrics, failures, trial durations and winner). CSV fields final_hpo_seconds,
final_hpo_trials and final_hpo_best_params report tuning independently of pipeline
search and final replay. Total downstream cost includes tuning plus final replay.
Old results without tuning are a different evaluation protocol and need reruns.
