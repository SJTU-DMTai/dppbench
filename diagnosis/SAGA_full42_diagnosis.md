| dataset | 起止行 | 候选 FAILED | 最终日志结果 |
|---|---|---:|---|
| amazon_beauty | 13–526 | 0 | 成功，std AUC 0.692698 |
| kuairec | 527–822 | 0 | 成功，std AUC 0.872318 |
| movielens | 823–1509 | 2 | 成功，std AUC 0.803379；部分候选 list 错误 |
| yelp | 1510–1938 | 5 | 成功，std AUC 0.885658；4 次 list、1 次空训练集 |
| elliptic_bitcoin | 1939–2098 | 47 | 冻结标签缺失，最终初始化失败 |
| beijing_air_quality | 2099–2157 | 0 | 合法性预检拒绝，初始化失败，无训练输出 |
| berka | 2158–3279 | 1 | 成功，std AUC 0.545001；非法采样参数 |
| bike_sharing | 3280–3338 | 0 | 合法性预检拒绝，初始化失败，无训练输出 |
| citibike_jc_hourly | 3339–3973 | 0 | 成功，std RMSE 0.824069；搜索 fitness 恒为 0 |
| default_credit | 3974–4673 | 3 | 成功，std AUC 0.501733；非法采样参数 |
| fraud_detection | 4674–5284 | 8 | 4 次空训练集、3 次非法参数、1 次内存超限；最终失败 |
| home_credit | 5285–5438 | 1 | 内存超限；最终失败 |
| nyc_taxi_hourly | 5439–5492 | 0 | 搜索启动前资源检查失败 |
| polish_bankruptcy | 5493–5544 | 0 | 搜索启动前资源检查失败 |

因此是 7 个 dataset 有最终指标、7 个最终失败。候选失败不能当作整个 dataset 失败，汇总中重复的 RuntimeError/BudgetExceededError 也不是新根因。

## A/B. 六类异常根因与一个静默评分错误

### 1. 多值类别与标量类别/整行哈希契约不匹配

- 影响：movielens（847、934 行），yelp（1527、1605、1710、1723 行）。
- 完整异常类型：`builtins.TypeError: unhashable type: 'list'`。
- 日志关键栈片段：`factorize -> factorize_array`，或 `unique -> unique_with_mask`；均在 JoinTable 后、下一个 DAG step 完成前。物理搜索的 TargetEncode 候选也出现 factorize 错误。
- 源码定位：`data_ctx.py::_profile_table/TableContext.categorical_cols` 丢弃 `RecData.col_types`；`label_encode.py::LabelEncode.transform` 默认对所有 object 列调用 `pd.unique`；`target_encode.py::TargetEncode.transform` 使用 `groupby(col)`；`deduplicate.py::Deduplicate.transform` 使用多列 `drop_duplicates`。
- 确定的源列：MovieLens loader 将 `item_genres` 按 `|` 分割为 list；Yelp loader 将 `item_categories` 分割为 list，均注册为 `categorical_list`。JoinTable 将这些侧表列带入交互表，日志却打印 `list_cols=0`。
- 原始截断日志无法逐一证明六个失败候选的完整 operator 参数；不能声称每次 factorize 都唯一对应 TargetEncode。源码和复现覆盖三条确定存在的不兼容路径。
- 分类：baseline schema 边界 + pipeline operator；不是 DIN 数值优化失败。
- 正确语义：保留 list，并由推荐数据层逐 token 编码；标量 LabelEncode/TargetEncode 不应把完整 list 当单一类别。Deduplicate 必须仍能比较这些列，因此只在临时比较键中转 tuple，输出仍为原 list，保留顺序与元素边界。没有全局 `str(list)`。

### 2. 数据依赖过滤把训练池变为空集

- 影响：yelp（1584 行）；fraud_detection（5013、5034、5049、5143 行）。
- 完整异常类型：`builtins.ValueError: cannot convert float NaN to integer`；`builtins.ValueError: With n_samples=0, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.`
- Yelp 证据：`FilterKCore: (3629,34) -> (629,34)` 后，CreateSequence/ReduceDimension 仍执行，最后 `_run_rec_training -> build_feature_columns` 失败。629 行是保留的 std-test 子集，搜索不将它加入训练特征构建，空整数列 `int(df[col].max())` 触发 NaN 错误。保留 std-test 的协议使这个潜在问题显现。
- Fraud 证据：`HandleMV: (4500,435) -> (1500,435)`，只余 held-out 部分，之后 `TabularData.split -> sklearn.train_test_split` 失败。HandleMV physical action=delete 对大量稀疏列删除所有训练行是合法但不可行的候选。
- 分类：pipeline operator 的数据依赖可行性与 evaluation 前置条件，不是测试集标签/模型本身的问题。
- 修复：在有 split 标记的 operator 输出边界校验训练池，并在最终 pipeline 输出再次校验；错误直接写明是哪个算子清空训练集。无标记的辅助支路允许暂时为空，避免破坏合法 left join。
- **不保证任意候选都成功**：空候选应被搜索记为失败，不能默默跳过过滤算子、降低 k、恢复已删除样本或拿测试样本训练。修复的是错误定位和训练前置条件，而非改变过滤定义。

### 3. Elliptic 旧冻结 schema 与通用标签验证顺序冲突

- 影响：elliptic_bitcoin，47 次同类候选失败。
- 类型：`builtins.ValueError: frozen-train is missing target column 'class'`；外层衍生 `builtins.RuntimeError: Could not generate any successful pipeline during initialisation.`
- 日志证据：`elliptic_bitcoin_data.py:198, load_std_test_frozen -> super().load_std_test_frozen()`；2047–2052 行为外层初始化异常。
- 源码链路：`TrainingExecutor._run_graph_training -> TabularData.run_pre_process/load_std_test_frozen -> labeled_rows`。旧 `train_frozen.parquet` 仅包含 features；旧 subclass 要在 super 返回后 attach labels，但 super 已因缺列抛错。
- 分类：std-test 文件 schema 兼容，不是 GNN 或 HPO 失败。
- 修复：增加冻结 frame 规范化 hook，在通用有限标签检查前，Elliptic 按 txId 从 classes 表补齐标签；未知节点保持 -1，保留图结构。不重建、不重抽样、不改 frozen membership。
- 回归验证：原始 `1/2/unknown` 映射正确；held-out 节点即使 time_step<=34 也只在 std_test_mask，不能进入训练/验证 mask。

### 4. JoinTable 参数生成错误要求存在 ID

- 影响：beijing_air_quality（2106–2111 行），bike_sharing（3287–3292 行）。
- 类型：`builtins.RuntimeError: Could not generate any successful pipeline during initialisation.`
- 日志保留的外层栈：`evaluate_baseline.py:762 main -> :481 _run_baseline_and_eval -> runner.run()`。没有候选训练栈，因为失败发生于 `LogicalSearch._evaluate -> check_pipeline`。
- 源码链路：`pipeline.py::make_step` 的 JoinTable 参数生成在 `ctx.id_col is None` 时返回 None；`repair/check_pipeline` 又要求带辅助表的候选包含 JoinTable，故初始化无合法候选。
- 分类：SAGA 使用的共享 baseline pipeline 构造逻辑。
- 修复：按实际 `join_keys_to_primary` 构造，而不要求键名是 ID。补齐 loader 声明的辅助表连接键：北京 station_meta 用 station；Bike day 用 dteday，hour/day 的 instant 是各自独立行编号，不能混用。避免从任意同名污染物列误推断 join key。
- 不改样本划分、标签和 benchmark protocol；连接键属于原始表的 schema 语义。

### 5. Oversample 搜索空间大于实际实现

- 影响：berka（2779）；default_credit（4342、4378、4399）；fraud_detection（5041、5069、5108）。
- 类型：`builtins.ValueError: method must be random/smote/adasyn`。
- 日志关键片段：`dataset.py:956 _run_pre_process_relation_dag -> _run_relation_dag`。
- 源码链路：`PhysicalSearch.tune_one -> get_param_space -> op_spec Oversample.param_space` 采到 `smote_nc`，`BaseData._run_relation_dag -> Oversample.__init__` 只支持 random/smote/adasyn。
- 分类：baseline/operator API 契约错误，与 final model HPO 无关（这是 pipeline physical search）。
- 修复：从共享搜索空间移除未实现的 smote_nc，保留全部已实现方法及执行步骤。没有把失败方法偷偷降级成其他方法。

### 6. 实际内存预算超限与对象生命周期

- 影响：fraud_detection、home_credit、nyc_taxi_hourly、polish_bankruptcy。
- 完整类型：`baselines.common.budget.BudgetExceededError`。
- 日志证据：fraud 全量训练已到最佳 iteration=2660，5212 行报 `76140.8MB/71680.0MB`；home 5365 行 `74141.8MB/71680.0MB`；后两项均在搜索前即 `71893.7MB/71680.0MB`。
- 关键外层栈：`evaluate_baseline.main -> _run_baseline_and_eval -> budget.check_resources -> BudgetExceededError`（5221–5226、5387–5392、5441–5446、5495–5500）。不是 torch CUDA OOM；原命令是 CPU。
- 源码发现：schema 推断后 cache 保留完整原始 dataset，训练又创建另一实例；DAG values 保存所有中间表；SearchPipelineEvaluator 闭包引用 data、data 又保存闭包，形成循环；HPO 的 evaluator/loader 也有 bound-method 引用链。原 cleanup 仅清理临时 YAML 目录，不释放数据 cache。
- 分类：execution/evaluation 资源生命周期 + 真实工作负载内存需求。日志不足以量化各项占比，也不足以证明全由一个泄漏引起。
- 修复：schema 提取后释放冗余 cache；DAG 按消费者计数释放已无用途的中间结果，共享支路仍保留到最后一个消费者；训练前及任务间 gc；cleanup 清 cache。Windows 预算采样改为当前 WorkingSetSize，由 Budget 自己累积 scope 峰值，避免继承进程历史峰值（Linux 日志已使用 VmRSS，此项不是该日志 Linux 故障的证明）。
- **未提高预算、未关闭监控、未跳过 full evaluation/HPO**。70GB 是否足够仍需服务器全量验证。单个任务仍真实超限时应继续失败并保留诊断。

### 7. 附加发现：回归 fitness 错读为 AUC 的 0 默认值

- 日志直接影响：citibike_jc_hourly，RMSE 各不相同而所有 `[eval] fitness=0.0000`，GA 不能按质量选优。beijing/bike 修复初始化后也会受此潜在缺陷影响。
- 无异常类型；属于 evaluation 评分错误。
- 根因：`PipelineEvaluator/SearchPipelineEvaluator` 默认 metric_key=auc，取不到 auc 时默认为 0；SAGA 实际用 LightGBM 回归模型。
- 修复：未显式传 metric_key 时读取所选模型配置的首个 train metric；RMSE/MSE/MAE/logloss 取负值供最大化搜索，原 metrics 保留正值。缺失或非有限指标判为失败，不制造 0 分。显式 metric_key 仍受尊重。

## 特别核查：std-test、推荐数据与 final HPO

- kuairec（529）与 movielens（825）显示 std-test protocol changed，并成功重建及最终评估；不能把 protocol changed 本身当错误。
- Elliptic 失败确与新通用 frozen 标签检查和旧 schema 的顺序冲突有关；Yelp/Fraud 空训练池与保留 held-out 的处理有关，但保留测试集是正确行为。
- `scripts/build_std_test.py::build_graph` 与 loader 已逐项检查；本次不修改其 holdout 规则或文件。
- `scripts/final_model_hpo.py::tune_final_model` 每 trial 都 `set_evaluate_std_test(False)`，按验证 metric 及方向选参，再创建新的最终 evaluator。通过 fake-evaluator 契约测试验证这一点；这不代替真实模型训练测试。
- 日志中的最终致命错误均在 runner/search 或 search budget 检查阶段，尚未进入该 dataset 的 final HPO。没有日志证据指向 final HPO 参数导致这些异常。
- HPO 多次 full-data replay 会放大资源压力，共享执行器生命周期修复也覆盖它；本次没有更改 HPO 空间、试验次数或验证/测试协议。
- 日志还存在单类 AUC UndefinedMetricWarning、常量特征/非有限值 RuntimeWarning、Berka 混合类型 DtypeWarning、pandas FutureWarning；它们不等同于本次七个最终失败。非有限 search metric 现在明确失败。Berka/Fraud 很高的 validation AUC 与低 std AUC 是值得后续检查的泛化/验证设计信号，不能仅凭此认定已证实泄漏，也不能认定本次修复解决了这些统计问题。

## C/D/E. 修改文件及正确性

| 文件 | 修改 |
|---|---|
| baselines/common/data_ctx.py | 保留推荐语义类型、区分 list/text/scalar；读取 loader 辅助连接键 |
| baselines/common/pipeline.py | JoinTable 使用有效连接键，不要求 ID |
| baselines/common/op_spec.py | Oversample 空间匹配实际方法 |
| dppbench/ operators/preprocessing/label_encode.py | 自动选择时排除 list/tuple/array，兼容 pandas string dtype |
| dppbench/ operators/cleaning/deduplicate.py | 用临时 tuple 键去重，原 list 原样返回 |
| dppbench/dataset.py | frozen schema hook、空训练集前置条件、DAG 中间结果生命周期 |
| dppbench/tasks/elliptic_bitcoin/elliptic_bitcoin_data.py | 验证前恢复 legacy frozen 标签 |
| dppbench/tasks/bike_sharing/bike_sharing_data.py | 声明 day 表按 dteday 连接 |
| dppbench/tasks/beijing_air_quality/beijing_air_quality_data.py | 声明 station_meta 按 station 连接 |
| baselines/common/executor.py | cleanup 清缓存、训练前回收循环引用、保留完整 traceback |
| baselines/common/evaluator.py | 从配置选指标、处理最小化及非有限值、清 schema 数据缓存、完整错误输出 |
| baselines/common/search_evaluator.py | 默认指标交由配置推断 |
| baselines/common/budget.py | Windows 使用当前 RSS 而非历史进程峰值 |
| baselines/SAGA/saga.py | 复用推断的 schema 并释放原数据缓存 |
| baselines/SAGA/logical_search.py | 初始化错误汇总实际候选原因 |
| scripts/evaluate_baseline.py | 完整外层 traceback、任务间回收循环引用 |
| baselines/SAGA/test_full42_regressions.py | 新增 14 项离线回归测试 |

修复集中在契约边界、资源生命周期及指标解释；没有新增吞异常逻辑，没有恢复失败算子之前的输入，没有通过改变测试集合使实验成功。多值类别保留内容，冻结集合成员不变，graph mask 阻止 held-out 进入监督训练。

## F. 测试与限制

通过：31 项（新增 full42 14 项、原 SAGA graph 2 项、BAT 5 项、DeepPrep 10 项）。覆盖 list schema/编码/去重，两个任务真实连接键，Oversample 方法契约，空训练池定位，legacy graph 标签及 mask，DAG 线性与共享分支存活期，默认 RMSE、最小化/NaN 处理、HPO 验证选择与最终独立 evaluator。17 个修改/新增 Python 文件通过 Python 3.9 语法解析和当前运行时 compile 检查；这不是 Python 3.9 真实执行测试。

测试命令（在配置好项目依赖的环境）：

```bash
python -m unittest baselines.SAGA.test_full42_regressions baselines.SAGA.test_saga_graph baselines.BAT.test_bat_alignment baselines.DeepPrep.test_deepprep_protocol -v
```

本地使用 Python 3.12 / pandas 3.0.1，原服务器日志为 Python 3.9；本地安装 PyYAML 到项目 `.test-deps` 运行测试。FlowPipe、DiffPrep 测试因缺少 PyTorch 导入失败，未执行断言。DagPipe/真实 DIN/GraphSAGE/LightGBM 全量训练没有运行。本地现有 pyarrow 不能被当前 runtime 正常导入，graph schema 测试对 parquet reader 使用内存 fixture，真实 loader 的标签规范化和 graph mask 逻辑仍被执行。无服务器全量数据，不能报告全量重跑通过。

## G/H. 修复状态与待验证项

- Elliptic 标签错误、Beijing/Bike 初始化错误：对应根因已修改且离线回归通过；实际 dataset 端到端仍待重跑。
- MovieLens/Yelp list 错误、Berka/Default/Fraud 非法 Oversample 参数：根因路径已覆盖；这些 dataset 多数原本已有最终指标，只是候选错误。
- Yelp/Fraud 空池候选：明确拒绝并定位到具体算子；不承诺这些无效候选会变成有效候选。
- Citibike 回归 fitness：修复且测试通过，应重跑，因为旧搜索结果并非按 RMSE 选出。
- Fraud/Home/NYC/Polish 内存失败：实现减持有量修复，但**不能宣称已恢复全量成功**；需在原机器验证峰值和最终指标。
- 完整历史 traceback 不可恢复；下次日志会输出完整栈。模型真实训练、跨 Python 版本及其他 torch baseline 仍需服务器验证。

## I. 推荐 SAGA 重跑命令

在原 Linux 项目根目录、原 dppbench 环境运行。使用全新唯一输出目录和日志，不覆盖旧实验；`--skip_build` 要求使用已有且版本匹配的 frozen std-test，缺失/过期时明确停止，不自动重写数据。保留原 seed、14 datasets、模型默认配置与预算；修复共享 schema 和 metric 后推荐全套重跑。

```bash
run_dir=$(mktemp -d /mnt/sdb/jiaqinyi/dppbench/output/exps/exp1_1_cost_effectiveness/seed_42_saga_fix_XXXXXX)
set -o pipefail
python scripts/evaluate_baseline.py \
  --final_hpo --baseline SAGA \
  --data_names amazon_beauty,kuairec,movielens,yelp,elliptic_bitcoin,beijing_air_quality,berka,bike_sharing,citibike_jc_hourly,default_credit,fraud_detection,home_credit,nyc_taxi_hourly,polish_bankruptcy \
  --data_dir /mnt/sdb/dengjiale/dppbench/data \
  --output_dir "$run_dir" \
  --output_csv "$run_dir/SAGA/results.csv" \
  --seed 42 --gpu_id -1 --skip_build \
  --max_pipeline_len 10 --max_evaluations 1000 \
  --max_memory_mb 71680 --max_gpu_memory_mb 81920 --max_llm_calls 100 \
  2>&1 | tee "$run_dir/SAGA_full42.log"
```

SAGA 不使用原命令中 CtxPipe/DeepPrep/Qwen 的 checkpoint 参数，故省略。上述命令只作为建议提供，本次未执行，没有删除或清理任何已有实验输出。
筛选文件


# exp1_fix_saga42.log 后续诊断

日志 54003 bytes、653 行，只运行到 elliptic_bitcoin 的 physical search。没有其余七个 dataset 的运行结果，也未进入 final HPO。原日志和已有输出未修改。

## 1. Input contains NaN：聚合缺失值进入 GNN

307–322 行的具体 pipeline：JoinTable → HandleMV → LabelEncode → AggregateGroupFeature。
聚合后由 169 列变为 175 列，图特征由 165 变为 171；epoch 1–10 全部 loss=nan。
323–331 行最终报告 `builtins.ValueError: Input contains NaN.`。

源代码链路：SAGA 默认 AggregateGroupFeature 使用 ctx.id_col=txId 分组，节点 ID 唯一；对 feat_* 的 sample std 使用 pandas 默认 ddof=1，因此单元素组生成 NaN。HandleMV 位于聚合之前，无法处理随后新增的缺失列。EllipticBitcoinData.build_graph 原先直接将所有 feat_* 列转 float32；NaN 进入 GraphSAGE，污染 logits/loss。训练结束后的 `_best_f1_threshold → sklearn.precision_recall_curve → assert_all_finite` 拒绝 NaN 预测值。这个最后调用链由当前源码定位；服务器日志仍截断了项目栈帧，不能声称日志保存了完整 traceback。

具体缺失列属于新生成的 `feat_*_std`；日志没有打印候选 agg_cols 参数，不能恢复准确的 feat 编号。离线使用真实 AggregateGroupFeature 和 build_graph 的复现验证该路径。

修复：保持 sample std 的统计语义（单元素组上游仍为缺失），在密集图模型输入边界插补。只使用监督训练 mask 内的非缺失值求中位数；训练列完全缺失时使用固定 0，避免使用验证/测试值估计。只作用于 feature matrix，不改标签、节点或 mask。Inf/float32 溢出明确报错，不偷偷替换。

新增图训练输入检查：训练/验证 mask 非空，验证 AUC 有两类，特征有限；非有限 loss 在出现的 epoch 立即报错，非有限预测在 metric/阈值计算入口明确报错，不再训练十轮后只显示 sklearn NaN。

## 2. 两次 val_auc=nan：split 标记因重排错位

156–176 行候选包含 CreateRollingFeature；285–305 行候选包含 SortRows。两者均在图构建后出现 `train=1386, val=0, test=0, std_test=689`，与正常候选的 `train=374, val=58, test=143, std_test=1500` 不同。不是单纯模型精度下降。

`TabularData._transform_tabular_dag_df` 原来移除 __split__，执行 operator，行数相同就按位置贴回原 marker。SortRows 和 CreateRollingFeature 内部都会重排行，行数不变并不代表样本顺序不变，导致 frozen-test 成员换位、时间 mask 异常。

修复：此分支按 train/val/test/std_test 分别调用算子一次，再给实际输出行附回对应标记。保持原集合成员归属；不再通过位置猜测，也不通过拼回测试样本修复训练。FIT_ON_TRAIN_ONLY 的已有训练拟合/后续 transform 分支保留。

## 3. 日志结尾：KeyboardInterrupt

602–653 行是 physical search 的 HandleMV(method=iterative) → IterativeImputer → BayesianRidge 被中断。它不属于 NaN、内存超限或 final HPO 失败。日志不能说明是谁发出了中断信号，不能认定运行自然崩溃。

本次不删除 iterative 候选、不把它替换成 median、不改变搜索预算；因此复杂插补候选仍可能较慢。

## 4. graph 搜索阶段仍输出 std-test 指标

executor 的 `_evaluate_std_test=False` 原先未传入 train_graph，后者总是计算 test/std-test 指标。虽然 fitness 取 val_auc，但测试指标不应在搜索/HPO trial 中计算和暴露。

新增 train_graph(evaluate_test=True)，executor 显式传入现有开关。搜索/HPO 保留图的 transductive 特征/结构，但不计算测试标签指标；最终评估默认仍计算测试指标。未改 HPO 空间或试验次数。

## 修改文件

- dppbench/dataset.py：修复重排时 split 标记错位。
- dppbench/graph_input.py：新增训练统计插补及图输入校验，独立于 torch。
- dppbench/tasks/elliptic_bitcoin/elliptic_bitcoin_data.py：在图模型边界调用特征规范化。
- dppbench/models/gnn.py：输入/loss/预测校验，尊重测试评估开关。
- baselines/common/executor.py：向图训练传递测试评估开关。
- baselines/SAGA/test_graph_nan_regressions.py：新增 6 项回归测试。

## 验证与重跑

37 项离线测试通过：新 graph NaN 6 项、此前 full42 14 项、SAGA graph 2 项、BAT 5 项、DeepPrep 10 项。6 个 Python 文件通过 Python 3.9 语法解析及当前运行时 compile 检查。没有 PyTorch，未进行真实 GraphSAGE 梯度训练，也未声称服务器全量通过。

```bash
python -m unittest baselines.SAGA.test_graph_nan_regressions baselines.SAGA.test_full42_regressions baselines.SAGA.test_saga_graph baselines.BAT.test_bat_alignment baselines.DeepPrep.test_deepprep_protocol -v
```

请把上述修改文件及新增 graph_input.py 一并同步到服务器实际运行目录。前次 executor 的完整 traceback 修改也必须同步：这份新日志仍从 `hon3.9/...` 半截开始，说明该运行输出仍含截断栈。

先沿用原命令参数，改为 `--data_names elliptic_bitcoin` 并使用新的输出目录/CSV，确认该任务端到端完成。`max_memory_mb=71680` 无需因这个 NaN 调整。之后再跑原计划剩余 dataset。不要覆盖已保存的旧结果。

# 再次诊断
SelectFeature 把构图必需的 txId 当成普通数值特征删掉了，因此 build_graph() 无法建立节点到边的映射。time_step 同样需要保护，否则会破坏时间划分。这是前面修复遗漏的图任务字段约束。
我会同时修正搜索生成的参数和数据执行边界：保留 txId、time_step、class，仍正常筛选特征列，不跳过 SelectFeature。