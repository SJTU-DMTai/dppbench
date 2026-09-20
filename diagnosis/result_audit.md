# 实验结果与代码审计

本次读取的是 `output/results.csv`（5 行）和 `output/results1.csv`（6 行），不是单数文件名。两份文件均为 SAGA，7 个 LightGBM 任务、4 个 DIN 任务；不包含 DagPipe 实验结果。未改动原 CSV，未加载数据集，未运行实验或模型测试。代码位置均指修改后的当前源码。

## 结果中的事实

**没有负的 ROC-AUC。** 9 个非空 `std_test_auc` 均合法，范围为 **0.5017330810–0.9704329402**；两个回归任务的 `std_test_auc` 为空。负数来自 `search_fitness/final_val_fitness/val_fitness`，其中误差指标被取负，便于统一最大化搜索；没有发现 Gini 或 `2*AUC-1` 变换。

| 文件/CSV 行号 | 任务 | 验证原始指标 | 独立测试指标 | 解释 |
|---|---|---:|---:|---|
| results / 2 | citibike_jc_hourly | RMSE 0.211484 | RMSE 2.549924 | CSV 的验证 fitness 是 -0.211484；测试误差约为验证的 12.1 倍 |
| results / 3 | fraud_detection | AUC 0.997864 | AUC 0.970433 | Oversample 后划分验证集，有确认的泄漏机制 |
| results / 4 | home_credit | AUC 0.999999893 | AUC 0.611759 | Oversample、JoinTable、聚合同时存在；测试输出 686,872 行值得核验 |
| results / 5 | nyc_taxi_hourly | RMSE 2.309151 | RMSE 44.660389 | CSV 的验证 fitness 是 -2.309151；测试误差约为验证的 19.3 倍 |
| results / 6 | polish_bankruptcy | AUC 1.000000 | AUC 0.908514 | 验证满分不能证明泛化满分 |
| results1 / 2 | amazon_beauty | AUC 0.785795 | AUC 0.692698 | 存在验证预处理泄漏机制，无法据差值量化贡献 |
| results1 / 3 | kuairec | AUC 0.871311 | AUC 0.872318 | 小幅反向差异本身不构成 bug |
| results1 / 4 | movielens | AUC 0.848204 | AUC 0.803379 | 需结合序列/滞后特征参数解释 |
| results1 / 5 | yelp | AUC 0.901973 | AUC 0.885658 | TargetEncode 在原验证切分前拟合 |
| results1 / 6 | berka | AUC 1.000000 | AUC 0.545001 | Oversample 后切分；不能把验证满分当有效评估 |
| results1 / 7 | default_credit | AUC 1.000000 | AUC 0.501733 | Oversample 后切分；测试输出 36,000 行，Join 存在展开机制 |

CSV 检查覆盖全部 32 列：非空数值无 NaN/Inf；没有 AUC 越界、负耗时、负计数、列数错误、重复的 task/baseline/model/output_dir 标识、算子数与 n_steps 不符。11 行错误字段均为空，但这不能排除旧版执行器吞掉错误。

以下一致/空白属于输出设计：`val_fitness=final_val_fitness`、`eval_time_s=final_eval_seconds`、两列 inference 时间相同；所有 HPO 都是 9 个候选；所有 baseline_time_kind 都是 search_inference，因此 baseline_training_seconds 全空；两个 RMSE 任务的 AUC 空白不应补零。所有预算 evaluations_used 比 n_unique_evals 多 1，不能仅凭此判断重复计费，二者分别是预算消费计数和所报告 evaluator 的缓存计数，需要运行日志确认其生命周期。

## 已确认并修改的问题

### 1. 验证 AUC 接近 1：划分发生在重采样和预处理之后

- **位置**：[dataset.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/dppbench/dataset.py:1153)、[executor.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/baselines/common/executor.py:518)、[base_model.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/dppbench/models/base_model.py:186)。
- **原因**：原表格流程是 run_pre_process → split。Oversample 先复制少数类，再随机划分，原样本及副本可同时进入 train/val。TargetEncode、缺失值填充、缩放等也先看到后来作为验证集的行。推荐模型同样在整个 DAG 之后切分，CreateSequence 先使用了未来被分配到验证集的交互。
- **正确逻辑**：原始样本固定 train/val/test，再训练拟合预处理及采样，验证/测试只应用训练得到的变换。
- **修改**：在数据层运行 DAG 前固定验证分区；沿用各任务原本的分层/时间划分；四个时间序列任务的 split 返回已固定分区。APPLIES_TO_STD_TEST=False 算子现在也跳过 val/test；FIT_ON_TRAIN_ONLY 算子先处理 train，再复用状态。推荐模型消费预先固定的标记；CreateSequence 只把 train 交互加入历史。训练入口补传模型配置，保留 val_ratio/seed。
- **证据边界**：5 个表格分类任务都列出了 Oversample，机制确认，但没有原始样本/保存的 pipeline 参数，无法算出每条结果有多少副本跨集合。

### 2. 测试行数可膨胀：JoinTable 的 key/rec 模式没有校验查表唯一性

- **位置**：[join_table.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/dppbench/%20operators/integration/join_table.py:133)、[default_credit_data.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/dppbench/tasks/default_credit/default_credit_data.py:105)。
- **原因**：文档声明 key 模式是 lookup，但 merge 接受重复右键。default_credit 的 monthly_history 明确为每个 ID 生成 6 行；直接 key join 会展开 6 倍。展开前后再随机切分，也会让同一个主体进入 train/val。对不等长的一对多关系，测试 AUC 还会隐式按辅助记录数加权。
- **正确逻辑**：一行预测对应一个原始评估样本；多条历史先按主体聚合，或明确设计预测聚合协议。
- **修改**：key/rec lookup 加 validate="many_to_one"；重复键明确报错。已有 method="agg" 可先聚合后连接。不擅自选择历史记录、去重或平均预测。
- **证据边界**：36,000 行与 6 倍展开机制吻合；CSV 没有 Join 参数和 frozen 样本数，不能断言这就是该次运行的实际路径。home_credit 同理。修复后不符合 lookup 约定的旧候选会失败，应改成明确的 agg 配置。

### 3. target 虽被模型删除，却可能通过派生特征重新进入 X

- **位置**：[dataset.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/dppbench/dataset.py:69)、AggregateGroupFeature.transform、ReduceDimension._select_cols。
- **原因**：AggregateGroupFeature 的 agg_cols=None 自动选择所有数值列，包含 target；按唯一 ID 分组时 target_mean 就是标签本身。ReduceDimension 的 target_col 默认 None，自动数值选择也可能包含实际标签。模型最后仅删除原标签，无法去掉其派生信息。
- **正确逻辑/修改**：数据层为默认聚合排除真实标签，拒绝显式用标签分组或聚合；为 ReduceDimension 绑定数据集真实 target，并拒绝把标签放进投影输入。
- **证据边界**：当前 common pipeline 已会拒绝部分显式 target 参数，但不能覆盖隐式默认；CSV 只记录算子名称，没有参数，不能确认这次运行命中了默认分支。没有猜测删除某个业务特征。

### 4. 行错位和重复索引

- **位置**：[oversample.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/dppbench/%20operators/preprocessing/oversample.py:69)、[undersample.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/dppbench/%20operators/preprocessing/undersample.py:71)、[create_rolling_feature.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/dppbench/%20operators/feature_engineering/create_rolling_feature.py:99)。
- **原因**：随机采样抽取 index 标签再 df.loc，在索引重复时一次可取回多行，甚至跨类别；groupby rolling 的结果顺序可能不同于输入，`.values` 回填会错配。
- **正确逻辑/修改**：随机采样前重置为唯一行索引；rolling 多聚合结果按 Series 索引对齐。SMOTE 的 X/y 在同一次 fit_resample 后按数组并排组装，未发现其自身错位。
- **证据边界**：CSV 中没有 CreateRollingFeature；这是静态发现的潜在 bug，不能归因于现有数值。

### 5. DAG/Concat 的分区隔离漏洞

- **位置**：[dataset.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/dppbench/dataset.py:350)、[concat_table.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/dppbench/%20operators/integration/concat_table.py:25)。
- **原因**：原框架对主输入按分区调用算子，但 secondary inputs 仍是完整 DataFrame。纵向 Concat 会把同一辅助表分别追加到训练/测试；已经带 __split__ 的 DAG 分支也可能被再次混入其他分区。
- **正确逻辑/修改**：纵向追加只用于训练分区；带分区的 secondary input 暂时明确拒绝，要求实现有明确定义的分区对齐，避免静默污染。普通无分区辅助表仍支持。
- **未覆盖的边界**：横向 Concat 仍按 pandas 索引对齐，调用者必须保证索引代表同一样本；本批结果没有 Concat。不能仅从代码推断某个辅助表是否包含未来信息或业务标签代理。

### 6. frozen test 存在但 frozen train 缺失时仍继续训练

- **位置**：[dataset.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/dppbench/dataset.py:680)、TabularData.load_std_test_frozen。
- **原因**：原实现只有在 frozen train 文件存在时才替换原全量训练数据；否则仍把测试行追加到包含它们的原始数据中。
- **修改**：std_test 存在时，配套 train_frozen/interaction_frozen 必须存在，否则立即报错。无法从现有 CSV 判断当时文件是否缺失；已存在的 frozen 文件是否真正不相交仍需数据核验。

### 7. 聚合/错误处理可生成误导性成功结果

- **位置**：[executor.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/baselines/common/executor.py:685)、[evaluate_baseline.py](/D:/智能软件所/2026/DAGpipe/dppbench-main/scripts/evaluate_baseline.py:578)、[common.sh](/D:/智能软件所/2026/DAGpipe/dppbench-main/scripts/exps/common.sh:185)。
- **原因与修改**：训练完成先置 success=True，后续测试报错时原 except 没恢复 False，现已修正；无标准测试主指标现明确报错；CSV 新增 val_metric_name/val_metric，保留旧 fitness 的方向约定。推荐验证原先各指标分别取历史最优，可能对应不同 epoch，现按保留的 checkpoint epoch 取值；静默训练 verbose=0 原先不计算验证历史，现验证计算独立于打印。
- **汇总**：CSV 写入直接按字典字段写值，没有 AUC 算术变换。shell 汇总原先递归收集所有 CSV，可能读取旧汇总甚至自身输出；现排除自身、已有 source_csv 汇总及缺少实验标识的文件。本批 11 行没有观察到重复实验。
- **补充**：删除了“验证历史为空时拿 test_results 当 validation fitness”的回退，测试异常不再只打印后继续成功。图任务 std_test_n_nodes→std_test_n_rows 原分支不可达，已修复，并禁止把 n_nodes 作为兜底指标；本批无图任务，属于静态确认的旁路错误。

## 完整链路核对与未确认风险

1. **数据处理**：分类任务标签通常为 0/1；Berka 从 status 映射后删除 status，Polish 辅助年份删除 class，这两处没有发现直接标签残留。无法检查真实数据值、标签代理和辅助表事件时间。
2. **split/重采样**：原始验证泄漏已修；std_test 本来通过 APPLIES_TO_STD_TEST=False 跳过 Oversample，不能说这批测试集被 Oversample 直接重采样。测试行数问题主要追踪 Join、过滤及构造协议。
3. **预处理/特征**：TargetEncode 现在只用训练分区统计。它对训练本身仍非 OOF，可能过拟合；这不再等于读取验证标签，未擅自改成另一编码算法。HandleOutlier 原先在验证划分前删除/修复样本，而测试保持原始，这会造成验证更容易，可解释回归误差差距的一部分机制，不能确定数值贡献。
4. **模型训练与预测**：X/y 从同一分区 DataFrame 提取；LightGBM 使用 X_val/y_val 评估，没有发现把 X_train 当作 X_val 的代码。LightGBM.predict_proba(X)[:,1] 对应 classes_[1]，与这里二元 0/1 标签及 sklearn 二元 AUC 的正类方向一致。DIN 经 PredictionLayer sigmoid 输出分数，预测 DataLoader 不打乱行序。未发现把 hard labels 当 AUC 概率、概率列反选或 pos_label 反转；不是 Gini。
5. **metric calculation**：TabularModel 和 BaseModel 均调用 sklearn roc_auc_score。TabularModel 已检查预测长度、有限值、单类标签。BaseModel 的 metric 异常仍返回 NaN，但最终评价器对主要指标做有限性检查；应保留错误而不能把 NaN 补零或裁剪。标准测试所有非主指标的有限性不一定都暴露在 CSV，原文件未包含完整指标字典，无法审计未输出的 MAE/R²/logloss。
6. **推荐任务协议**：_remap_ids/build_feature_columns 仍会参考多个分区的 ID/词表；这属于测试特征域参与构建，严格归纳协议需要 train-only vocab 和 OOV 约定。不能直接删测试 vocab，否则可能引入 embedding 越界；本次未猜测性重构。部分 FilterKCore/HandleNonIID 允许按训练域过滤测试用户/物品，可能改变固定测试人群；是否允许必须由评估协议决定。
7. **时间序列**：当前原始 split 是时间划分。目标 lag/rolling 是否允许消费更早测试时刻的真实标签取决于滚动单步预测还是固定起点多步预测。没有协议及具体 pipeline 参数，未武断禁用或改写。
8. **DagPipe**：当前独立适配器在 training_splits 中先固定分区；InputEncoder 只拟合训练，FrozenTransform 用固定训练参考逐查询变换，再返回原分区。未看到与公共旧路径同样的“先采样后划分”。本批 CSV 没有 DagPipe，不能拿 SAGA 数值评价它。
9. **耗时口径**：final_model_training_seconds 当前是 final_eval 总时间减一次测试 evaluate 时间，包含数据读取、预处理、验证等；inference_seconds 也包含指标计算，DIN 还重复执行过测试评价。这是已确认的命名/测量口径问题，不是纯 model.fit/predict 时间。未改变历史列含义；比较性能时应将其视作端到端近似耗时。Yelp HPO 162,739.6 秒约 45.2 小时，不能单凭其大认定计时错误。
10. **历史归因**：results 输出路径为 exps_fix，results1 为 exps；CSV 没有源码 commit、best_pipeline_yaml 和 split 指纹。当前源码 bug 可以确认，不能证明当前文件就是当时执行版本，也不能离线“修正”现有 AUC。

## 验证与后续复现实验条件

- 已用仅依赖标准库的 scripts/audit_result_csv.py 重读 CSV、检查数值/别名/行结构，生成 output/result_audit.json。
- 对 dppbench、baselines、scripts 共 258 个 Python 文件完成 AST 语法检查；没有导入任务加载器，没有触发下载，没有执行任何数据集实验或测试。语法通过不代表训练运行时已经验证。
- 现有结果保留作历史记录，验证 AUC 不应再用作方法有效性的主要证据。修复无法从汇总表逆推出可信新分数。
- 有数据后优先核验：原始样本 ID 的 train/val/test 不相交；重采样副本只在 train；Join 前后每个评估样本的行数；frozen 计数和标签不变；保存完整 YAML、源码版本、样本 ID/预测/标签及各阶段行数。再重跑搜索及 HPO，不能只换最后一次测试计算。