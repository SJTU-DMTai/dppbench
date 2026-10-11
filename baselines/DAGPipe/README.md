# DAGPipe task adapter

Numerical safeguard: RobustScaler uses unit scale for minibatch columns with
IQR at or below epsilon. This prevents tied categories from being amplified by
1e8; the safeguard applies to both search and frozen-transform evaluation.

This adapter learns the released hierarchical DAGPipe feature transform with a
supervised MLP surrogate, then freezes it before training the benchmark's selected
downstream model. Recommendation and graph support are DPPBench extensions of
the tabular implementation, not claims of an original paper implementation for
these task types.

## Supported datasets

All 14 task directories shipped with this repository are supported:

| Task type | Datasets | Downstream models |
| --- | --- | --- |
| Tabular classification | berka, default_credit, fraud_detection, home_credit, polish_bankruptcy | Task-configured tabular models |
| Time-series regression | beijing_air_quality, bike_sharing, citibike_jc_hourly, nyc_taxi_hourly | Task-configured tabular/forecasting models |
| Recommendation | amazon_beauty, kuairec, movielens, yelp | FNN, DeepFM, DIN, DIEN, SIM |
| Graph node classification | elliptic_bitcoin | GCN, GraphSAGE, GAT |

`tenrec` is an existing entry in the shared registry, but this checkout does not
contain a Tenrec loader or task configuration; it is not a shipped dataset.

## Recommendation

The adapter loads `interaction_frozen.parquet`, fixes train/validation membership,
and joins user/item side tables with many-to-one validation. It builds the
configured histories using training interactions only; neither validation nor
standard-test interactions extend another row's history. All sequences use the
saved sequence length and reference-column mapping, including Amazon Beauty's
brand history.

DAGPipe learns dense features from scalar interaction/side attributes and history
lengths. Separate user/item/reference IDs and sequences are preserved for the
downstream recommendation model. Their saved vocabularies are fitted only on the
training partition, start at 1, and map unseen/missing values to 0. The configured
target is never an input and its labels are preserved.

The checkpoint includes this schema and split configuration. Final HPO changes
model hyperparameters without refitting the DAGPipe encoder, ID maps or weights.
Task adapters count towards `max_pipeline_len`: one combined side-table join
(when needed), one operation per sequence, and one recommendation encoding step,
in addition to input encoding and learned operations. The default experiment
limit of 10 accommodates all shipped tasks. Too-small limits raise an error.

## Graphs

Elliptic's chronological supervision masks are retained: labeled nodes through
time step 34 train the transform, steps 35–39 select it. Unknown-label nodes and
future nodes do not contribute supervised training loss. The search surrogate
uses node attributes with an MLP; it does not learn graph topology. Final GNN
training uses the original graph edges and the transformed `feat_*` attributes.
Node identifiers, times, labels, and unknown nodes are preserved.

Frozen standard-test nodes are excluded from train/validation masks. Final
evaluation reports `std_test_auc`, `std_test_f1`, and `std_test_n_rows` through the
shared GNN executor. Search and final HPO do not read `std_test.parquet`.

## Running

Use the normal evaluation entry point; it prepares missing frozen splits:

```bash
python scripts/evaluate_baseline.py --baseline DAGPipe \
  --data_names amazon_beauty,kuairec,movielens,yelp,elliptic_bitcoin \
  --data_dir /path/to/data --output_dir /path/to/output \
  --max_pipeline_len 10 --final_hpo --seed 42 --gpu_id -1
```

To run all shipped datasets, pass
`--data_names berka,default_credit,fraud_detection,home_credit,polish_bankruptcy,beijing_air_quality,bike_sharing,citibike_jc_hourly,nyc_taxi_hourly,amazon_beauty,kuairec,movielens,yelp,elliptic_bitcoin`.
Explicitly select these tasks because the shared registry also contains the
unimplemented Tenrec entry. Direct `DAGPipe.run()` callers must first create the frozen training artifacts using
`scripts/build_std_test.py`. Missing artifacts fail explicitly rather than
falling back to training on the complete raw dataset.

## Tests and limits

If HPO reports `Unsupported pipeline type: DAGPipePipeline`, the checkpoint has
reached the public operator-DAG validator. This is an evaluator integration
error, not a dataset-format error. The DAGPipe evaluator now owns its constructor
and evaluation loop, directly instantiates `DAGPipeExecutor`, and directly calls
its checkpoint validator. It does not call the common evaluator's constructor
or `evaluate()`, including on installations with older non-overridable versions.
It still preserves caching, budgets, validation metrics, and frozen-test replay.
The HPO log and `final_hpo.json` record the evaluator class and source file to
help identify stale or unexpected imports on a remote installation.

```bash
python -m pytest baselines/DAGPipe/test_adapter.py baselines/DAGPipe/test_task_support.py -q
```

Tests use synthetic observations with real task classes, configurations, split
methods, downstream models, and evaluator/HPO code. Raw loading is substituted
for the per-dataset contract fixtures; the bike and graph integration fixtures
also exercise real CSV loading. They verify all 14 dataset interfaces, all five
recommendation models, all three GNNs, unknown IDs, sequence histories, graph
masks, checkpoint round-trips, and isolation of held-out data. These tests do not
measure performance or accuracy on the complete external datasets.

The inherited frozen transform evaluates each row against a fixed training
reference to keep predictions independent of other queries. This can be
expensive on large datasets; full-data runtime has not been benchmarked here.

## Runtime diagnosis and recovery after failed final HPO

Run in the same environment and checkout as the experiment:

```bash
python scripts/check_dagpipe_runtime.py --data_names amazon_beauty,kuairec
```

This prints imported evaluate/validator/executor/pipeline source paths and SHA-256
hashes, then exercises legal and wrong-task synthetic checkpoints. It reads task
configuration only and uses a disposable temporary directory, without loading
datasets or training models. Both tasks must report `status: passed`. This checks
dispatch compatibility, not full-data feasibility. The evaluation harness also
runs this probe before DAGPipe search or recovery.

Synchronize the whole `baselines/DAGPipe` directory together with
`scripts/evaluate_baseline.py`, `scripts/final_model_hpo.py`, and
`scripts/check_dagpipe_runtime.py`. Changing only the registry class name cannot
fix an old evaluate implementation.

If search saved `best_pipeline.yaml`, reuse it without repeating search:

```bash
python scripts/evaluate_baseline.py --baseline DAGPipe --final_hpo \
  --data_names amazon_beauty,kuairec --seed 42 --gpu_id -1 \
  --data_dir /mnt/sdb/dengjiale/dppbench/data \
  --resume_pipeline '/mnt/sdb/jiaqinyi/dppbench/output/exps_am_ku/exp1_1_cost_effectiveness/seed_42/DAGPipe/{task}_{model}/best_pipeline.yaml' \
  --output_dir /mnt/sdb/jiaqinyi/dppbench/output/exps_am_ku/exp1_1_cost_effectiveness/seed_42_dagpipe_replay \
  --max_pipeline_len 10 --skip_build
```

`{task}` and `{model}` are substituted for each task. Recovery requires a matching
checkpoint task and seed; a missing checkpoint fails instead of starting search.
Use the original frozen training and test files. Recovery disables rebuilding
missing/stale test splits. Old checkpoints have no dataset-content fingerprint:
they cannot prove those files have remained unchanged since search. If they
changed, perform a fresh run with a consistent split.

Use a separate output directory to preserve the original experiment. Recovery
writes `resume.json`, a checkpoint copy, `final_hpo.json`, and the results CSV.
The `pipeline_reused`, `pipeline_source`, and `pipeline_sha256` fields identify
recovery. Search duration, fitness, and evaluation count are unknown/blank, not
zero. HPO and final evaluation durations describe only the current run. Retain
original logs separately when computing complete experiment costs.

Run all regression tests, including real downstream recovery on synthetic data:

```bash
python -m pytest baselines/DAGPipe -q
```
