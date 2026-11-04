# Normative v4 protocol

This is a DPPBench adaptation of the released hierarchical weighted DAG, not
an exact author reproduction. It does not target an AUC or a dataset. New
training rejects old operator-space configurations; old checkpoint replay
continues through the unchanged legacy dispatch and scaling protocols.

## Candidate space, schema, and step count

`protocol_v4.CANDIDATES` is the single active registry. Its ordered stages are:

1. Numeric imputation: MeanImputer, MedianImputer (numeric/date channels only).
2. Categorical encoding: TargetEncoder when eligible, FrequencyEncoder.
3. Numeric transforms: Identity, StandardScaler, RobustScaler, MinMaxScaler.
4. A second copy of the numeric transform stage.

Stages with no applicable columns are omitted. Category-only inputs also offer
Identity in their encoding stage. No continuous numeric feature is frequency
or target encoded. Explicit categorical schema wins over numeric storage dtype;
otherwise strings are categorical and numeric columns remain numeric.

After counting existing task adapters and one DAGPipeInputEncoding node, nodes
are allocated round-robin: one per applicable stage, then the next candidate in
each stage until the remaining budget is exhausted. A budget too small to cover
the applicable stages fails explicitly. The limit remains <=10. Identity and
repeated operators in different layers EACH consume one step. Unselected
candidates do not execute. Nodes are not pruned/renamed to conceal work.

For budget 10 with no task adapter:

- Numeric only: [Mean,Median], [Identity,Standard,Robust,MinMax],
  [Identity,Standard,Robust] = 9 nodes + encoding.
- Batch mixed numeric/category: [Mean,Median], [Target,Frequency],
  [Identity,Standard,Robust], [Identity,Standard] = 9 nodes + encoding.
- Temporal mixed: the numeric-only node allocation, with categorical channels passed through.
- Category-only temporal: [Identity] + encoding = 2 steps.

Input encoding casts numeric/datetime values and transports deterministic
ordinal category IDs (missing/unknown=-1); it DOES NOT normalize or fill numeric
data. Mean=0, scale=1, fill=0 in its transport metadata are enforced by checkpoint
validation. Actual imputation runs in counted nodes, even when search training
has no missing values, because deployment may have missing inputs. All-missing
numeric training columns fail: public mean/median cannot fit a finite value.

Not enabled, with reasons printed before search:

- KNNImputer: sklearn nan-euclidean KNN/public all-missing behavior is not the
  vendor soft-KNN operation. A verified differentiable adaptation is absent.
- QuantileScaler: vendor per-batch ranks are not a verified public transformer.
- LogScaler: vendor train-min offset plus clamp is not a public log contract.
- PolynomialTransform: vendor elementwise squaring differs from public appended
  polynomial features/interactions. Appending changes the fixed-width router
  contract. This is a structural exclusion, not an AUC/numerical blacklist.

## Public semantics and differentiability

Mean/median nodes compute the public numeric missing-value statistic on all
training reference source values, fill only NaNs, and then apply routing. Their
non-numeric channels pass through. Scalar numeric transformations operate on
their routed current numeric channels. Non-numeric channels pass through, so
no scaler statistic can incorporate target-encoded categorical channels.

Standard uses sample std and public EPS=1e-6 (std below EPS -> denominator 1).
Robust uses interpolated median/Q25/Q75, keeping EVERY positive IQR, with zero
IQR -> denominator 1. MinMax uses [0,1], constant columns output 0 even for a
new query value. Tensor float64 is retained for gradients; public DataFrame
float32 output is compared with explicit tolerances. No quantile/log/polynomial
formula is silently substituted. Public source files are not modified.

Public TargetEncode and FrequencyEncode actually fit the maps. The fixed-width
branch replaces categorical channels, whereas public DataFrame operators append
columns: matching encoded VALUES is an explicit width-preserving adaptation,
not whole-DataFrame equivalence. Maps are non-differentiable frozen state;
gradients still train routing and connection weights. Public scalar numerical
formulas are expressed in Tensor operations and checked against public outputs.

## Labels and chronological safety

TargetEncoder is eligible for batch binary/regression tasks with at least two
nonempty feature-hash folds. A fixed SHA256 of canonical float64 encoded feature
bytes plus seed selects one of five folds (configurable >=2). Labels, batch
membership and row order are never inputs to the fold hash. Duplicate rows
share a fold. For fold f, maps AND the smoothing global mean use only legal
training rows outside f. A training row cannot contribute its label to its own
map or prior. The same fold-dispatch rule is used at deployment; there is no
full-training-map refit that would change the protocol for validation/test.

Every row performs one fold-selected lookup. The multiple stored tables are
fitted parameters of that node, not extra hidden executed DAG nodes. Missing
or unseen categories receive the fold's public global-mean fallback. Frequency
uses the full legal training category distribution, with public fallback 0.
Both maps persist in the checkpoint and are independent of query batch size.

Both categorical encoders are disabled for recommendation and graph tasks
(conservatively), and when the schema declares a sort column, timestamp or
datetime type, or adapter time field. A training-only frequency table can still
use events after the row being encoded. Event-time visibility has not been
established for those tasks. Multiclass scalar target means are also disabled.

OOF prevents direct own-label leakage in the encoding, not ordinary supervised
learning of model parameters on training labels. Existing canonical split and
task adapters remain responsible for training membership. No held-out label
enters fitted maps, numeric statistics, or the proxy training loss.

## Search, reference fitting, freezing and replay

A single seeded permutation selects up to 16 legal training reference rows
without replacement. The same sample is used throughout optimization and freezing.
Category maps on eligible batch tasks still use the legal training partition.
At every node its routed/transformed prefix provides statistics at CURRENT DAG
weights. Query/validation rows do not contribute. Gradients propagate through
those numeric statistics and the original route/connection/aggregation graph.
The MLP proxy and DAG jointly use Adam, the existing validation-loss selection,
sparsity penalty and early stopping. Upper-triangular adjacency is acyclic by
construction; its mathematical acyclicity penalty is zero.

After selecting the best weights, one no-gradient training-reference pass
freezes every node's imputation values/center/denominator/constant mask. These
statistics and category maps are serialized. A validation-feature probe checks
search-vs-frozen outputs at rtol=1e-10, atol=1e-12 before export. It uses NO
validation labels for fitting. Frozen execution evaluates only query rows using
saved statistics and the same graph and weights; it never recomputes statistics
from a query batch or reruns the training reference for every query batch.

Checkpoint v4 stores schema, allocation, budget, seed, maps, sampled training
reference (audit/provenance; no labels), training row count, weights and frozen
node statistics. Validation checks format/typed space, shapes, finite state,
positive denominators, category maps, fold seed, budget and exact allocation.
This is not a cryptographic attestation of data membership. Upstream frozen
data provenance still matters. Old v2/v3 checkpoints are not upgraded/relabelled.

The reference cap bounds repeated reference computation and activation storage.
Sampling may miss observed values in sparse columns. If a numeric column has
no observations in the chosen sample, search fails explicitly. Small positive
IQR remains possible. Earlier v4 full-reference checkpoints remain readable
under their recorded fit protocol.

## Numerical diagnostics, fairness and reproducibility

Float64 is required. Infinity in raw numeric data is rejected; it is not clipped
or silently imputed. All-missing numeric fit columns, nonfinite node outputs,
nonfinite layer aggregates, losses and gradients fail explicitly. Large FINITE
values are retained. Per-node observed absolute maxima are saved in the search
log; public small-IQR behavior can still produce 1e8 values. No AUC-driven
epsilon, clipping or dataset exception is introduced.

The common budget/evaluator/downstream model/metrics remain unchanged. Costs
include proxy train/val forwards (including ALL forwarded reference rows),
backward/optimizer calls, category table fits and fold fit row counts, final
statistic fitting and freeze verification. Inclusive host timings must not be
summed across nested operations. Existing total search timing includes input
preparation and checkpoint serialization. HPO/test fitting remains outside
search, controlled by the harness. Search logs record library versions, device,
candidate registry, node peaks and the freeze equality check.

Seeds initialize Python/NumPy/Torch and the permutation generator. Fold hashes
are deterministic. CPU repeat training is tested for equal serialized results.
GPU/platform bitwise equality is NOT claimed; run the acceptance suite in the
actual server environment. Old result CSVs and checkpoints are left untouched.

## Tests and server acceptance

```bash
python -m unittest baselines.DAGPipe.test_protocol_v4 scripts.test_dagpipe_public_semantics scripts.test_diagnose_dagpipe_nodes scripts.test_dagpipe_same_data -q
python -m pytest baselines/DAGPipe -q

CUDA_VISIBLE_DEVICES=2 python -m baselines.DAGPipe.acceptance \
  --task elliptic_bitcoin \
  --data-dir /mnt/sdb/dengjiale/dppbench/data \
  --device cuda:0 --seed 42 --epochs 1 \
  --output-dir outputs/dagpipe_v4_acceptance_elliptic_seed42
```

Use a new output directory. Acceptance does one search epoch, freeze and reload,
with budget=10 and no downstream training, HPO or standard-test evaluation.
Return `acceptance.json`, `DAGPipe_search.json`, `search_cost.json` and stderr if
it fails. Repeat for a real categorical/rec task before formal experiments.
Acceptance reports both the legal training row count and the sampled reference row count.

Local synthetic coverage: public scaler/imputer values, own-label and own-fold
label exclusion through the entire DAG, categorical-only/temporal applicability,
unknown categories, NaNs/all-missing/infinity, finite router/edge gradients,
budget allocation/checkpoint validation, batch/permutation invariance, frozen
query-only execution, graph training membership, actual two-epoch runner,
cost counters and identical repeated CPU checkpoints. Existing v2/v3 replay
tests remain included. PyArrow/pytest and real GPU/data integrations need the
server environment when unavailable locally.

All DAGPipe task/seed runs used as the formal v4 benchmark must be rerun from
search through final evaluation/HPO according to the unchanged experiment
protocol. Reusing old weights is not a v4 experiment. Other Baselines do not
need reruns because this change does not modify shared operators or their
algorithms. Archive/report old DAGPipe results under their original version.
