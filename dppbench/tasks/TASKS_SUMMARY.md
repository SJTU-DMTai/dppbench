# dppbench 任务总览（Tasks Summary）

> 路径：[`dppbench/tasks/`](file:///Users/bytedance/Documents/dppbech/dppbench/tasks)
> 本文件由对当前 `tasks/` 目录的全量扫描自动整理，包含每个任务的输入表数量、特征量级、预处理算子数量及综合复杂度评估。

---

## 1. 总览

| # | 任务名 | 数据基类 | 模型 | 任务类型 | 主表 + 辅助表 | 算子数 | 复杂度 |
|---|---|---|---|---|---|---|---|
| 1 | [amazon_beauty](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/amazon_beauty) | RecData | DIN | 序列推荐 | 1 + 1 = 2 | 6 | ★★☆☆☆ |
| 2 | [beijing_air_quality](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/beijing_air_quality) | TabularData | LightGBM | 时序回归（PM2.5） | 1 + 1 = 2 | 11 | ★★★☆☆ |
| 3 | [berka](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/berka) | TabularData | LightGBM | 二分类（贷款违约） | 1 + 7 = 8 | 17 | ★★★★★ |
| 4 | [bike_sharing](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/bike_sharing) | TabularData | LightGBM | 时序回归（cnt） | 1 + 1 = 2 | 12 | ★★★☆☆ |
| 5 | [bondora](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/bondora) | TabularData | LightGBM | 二分类（违约） | 1 + 1 = 2 | 13 | ★★★☆☆ |
| 6 | [citibike_jc_hourly](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/citibike_jc_hourly) | TabularData | LightGBM | 时序回归（小时租量） | 1 + 0 = 1 | 10 | ★★★★☆ |
| 7 | [default_credit](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/default_credit) | TabularData | LightGBM | 二分类（违约） | 1 + 1 = 2 | 11 | ★★★☆☆ |
| 8 | [elliptic_bitcoin](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/elliptic_bitcoin) | TabularData (Graph) | GCN/SAGE/GAT | 节点分类（非法/合法） | 1 + 2 = 3 | 8 | ★★★★☆ |
| 9 | [fraud_detection](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/fraud_detection) | TabularData | LightGBM | 二分类（欺诈） | 2（load 内 merge） | 9 | ★★★★☆ |
| 10 | [home_credit](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/home_credit) | TabularData | LightGBM | 二分类（违约） | 1 + 6 = 7 | 19 | ★★★★★ |
| 11 | [movielens](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/movielens) | RecData | DIN | 序列推荐 | 1 + 2 = 3 | 6 | ★★★☆☆ |
| 12 | [nyc_taxi_hourly](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/nyc_taxi_hourly) | TabularData | LightGBM | 时序回归（小时单数） | 1 + 0 = 1 | 12 | ★★★★☆ |
| 13 | [polish_bankruptcy](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/polish_bankruptcy) | TabularData | LightGBM | 二分类（破产） | 1 + 4 = 5 | 10 | ★★★★☆ |
| 14 | [tenrec](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/tenrec) | RecData | DIN | 序列推荐（点击） | 1 + 0 = 1 | 2 | ★☆☆☆☆ |
| 15 | [yelp](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/yelp) | RecData | DIN | 序列推荐（评分） | 1 + 2 = 3 | 6 | ★★★☆☆ |

> 备注：[`ifasion/`](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/ifasion) 目录仅包含原始 `.txt`（user/item/outfit），尚未实现 `data.py` 与 `pre_process.yaml`，因此**未纳入正式任务列表**。

按数据/模型类型分布：

- 表格二分类（6 个）：berka、bondora、default_credit、fraud_detection、home_credit、polish_bankruptcy；模型族覆盖 `LightGBM`、`MLP`、`TabTransformer`、`FTTransformer`、`SAINT`
- 表格时序回归（4 个）：beijing_air_quality、bike_sharing、citibike_jc_hourly、nyc_taxi_hourly；模型族覆盖 `LightGBM`、`LSTM`、`Transformer`
- 序列推荐（4 个）：amazon_beauty、movielens、yelp、tenrec；模型族覆盖 `FNN`、`DeepFM`、`DIN`、`DIEN`、`SIM`
- 图神经网络（1 个）：elliptic_bitcoin；模型族覆盖 `GCN`、`GraphSAGE`、`GAT`

### 1.1 本轮 pre_process 配置优化说明

本轮对现有 `pre_process.yaml` 做了面向 std-test 自动验证的保守优化，并新增两个任务目录脚本：

- [`audit_preprocess_configs.py`](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/audit_preprocess_configs.py)：审计 YAML、算子、列引用、泄漏风险和 schema 兼容性，输出 `PREPROCESS_CONFIG_AUDIT_REPORT.md`。
- [`evaluate_task_model_matrix.py`](file:///Users/bytedance/Documents/dppbech/scripts/evaluate_task_model_matrix.py)：构建/复用 std-test，执行预处理和模型训练；支持 `--models default|all|...` 与 `--task-family` 做 task × model 矩阵验证，输出 task-model matrix Markdown/JSON/CSV 指标。

关键修复点：

- `bike_sharing`：删除 `casual`、`registered` 目标组成项，并在 `day` 辅助表聚合前删除 `cnt/casual/registered`，避免目标泄漏。
- `citibike_jc_hourly`：移除 `birth_year` 清洗和聚合，兼容 2023 JC Citi Bike schema；保留 trip duration 清洗、小时重采样、lag/rolling 与排序。
- `beijing_air_quality`：`pollutant_sum` 改为不含当前 `PM2.5` 的 `co_pollutant_sum`，并在 station metadata 聚合前删除 `PM2.5`，避免即时目标泄漏。
- `fraud_detection`：移除训练集过采样，保留原始欺诈比例，用 LightGBM AUC 在 std-test 上评估，避免 590k×430 宽表被随机过采样放大到百万行级别。
- 推荐任务：`amazon_beauty`、`movielens`、`tenrec`、`yelp` 补充目标/ID 非空过滤；`amazon_beauty` 额外删除高噪文本名列并填补 `vote/item_price`；`tenrec` 在 fast 验证中使用 100 万行读取上限，full 模式仍可走完整 120M 行源文件。

### 1.2 模型矩阵扩展与验证结果

所有正式任务的 `model.yaml` 已接入向下兼容的 `model_options`，训练入口支持用 `--model` 或验证脚本的 `--models` 灵活选择模型。模型矩阵 fast 验证覆盖 65 个 task × model 组合，全部完成预处理、训练、评估与 std-test 指标落盘：

| 任务族 | 任务数 | 模型 | 验证组合 |
|---|---:|---|---:|
| 推荐 | 4 | `FNN`, `DeepFM`, `DIN`, `DIEN`, `SIM` | 20 |
| 表格二分类 | 6 | `LightGBM`, `MLP`, `TabTransformer`, `FTTransformer`, `SAINT` | 30 |
| 时序回归 | 4 | `LightGBM`, `LSTM`, `Transformer` | 12 |
| 图节点分类 | 1 | `GCN`, `GraphSAGE`, `GAT` | 3 |

验证产物：

- 汇总报告：[`MODEL_MATRIX_VALIDATION_REPORT.md`](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/MODEL_MATRIX_VALIDATION_REPORT.md)，结果为 `65/65 success`，总耗时约 2893 秒。
- 机器可读结果：[`model_matrix_validation_results.json`](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/model_matrix_validation_results.json) / [`model_matrix_validation_results.csv`](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/model_matrix_validation_results.csv)。
- 分族验证报告：`MODEL_MATRIX_REC_REPORT.md`、`MODEL_MATRIX_TABULAR_REPORT.md`、`MODEL_MATRIX_TIMESERIES_REPORT.md`、`MODEL_MATRIX_GRAPH_REPORT.md`。

---

## 2. 详细对比表

### 2.1 表格类（TabularData）

| 任务 | 主表行数级别 | 辅助表 | 特征数（处理前 → 处理后大致） | 算子链 | 关键脏数据点 |
|---|---|---|---|---|---|
| beijing_air_quality | ~420k 行（12 站 × 35064h） | `station_meta`（站点维表，1） | 11 → ~30+ | CreateFeature×2 → TransformPower → ScaleFeature → CreateLagFeature → CreateRollingFeature → JoinTable → CustomProcess → LabelEncode → HandleMV | 多站时序、滞后/滚动衍生 |
| berka | ~6.2k loan | account, client, disp, card, order, trans, district（7 表） | 6 → ~50+ | ParseDate×5 → OrdinalEncode → CastType → CreateFeature → TransformPower → JoinTable×4 → CustomProcess → CrossFeature → LabelEncode → HandleMV | 多表 ASCII；YYMMDD 整数日期；状态映射（A/C→0, B/D→1） |
| bike_sharing | 17 379 小时 | `day_aux`（日级聚合表） | 16 → ~30+ | ExtractDateTimeFeature×2 → CustomProcess(day drop target parts) → CreateFeature → TransformPower → JoinTable → CreateLagFeature → CreateRollingFeature → CustomProcess×2 → LabelEncode → HandleMV | 时序滞后/滚动；日聚合特征；删除 `casual/registered` 防泄漏 |
| bondora | ~50k loans | `country_stats`（loader 内 groupby 派生） | ~110 → ~50（列过滤后） | CustomProcess → CreateFeature×2 → TransformPower → ScaleFeature → JoinTable → HandleError → CustomProcess×2 → LabelEncode → HandleMV → ScaleFeature → Undersample | xlsx；高基类别；国家异常值修复 |
| citibike_jc_hourly | ~140 万行原始 trip → resample 到小时桶 | 无 | 11 → ~30+ | HandleOutlier(action=delete) → HandleError(action=delete) → ResampleTimeSeries → ExtractDateTimeFeature → CreateLagFeature → CreateRollingFeature → CustomProcess → LabelEncode → HandleMV → SortRows | 兼容无 `birth_year` schema；离群点；不规则时间序列重采样 |
| default_credit | 30 000 行 | `monthly_history`（wide→long 派生） | 24 → ~40+ | CustomClean → CreateFeature → TransformPower → JoinTable → CustomProcess → OneHotEncode → LabelEncode → HandleMV×2 → ReduceDimension → SelectFeature | EDUCATION/MARRIAGE 异常值；月度历史 wide→long pivot |
| elliptic_bitcoin | 203 769 节点 | `classes`（标签）+ `edges`（234 355 条） | 165 节点特征 → Z-score 165 维 | JoinTable → CustomClean → Deduplicate → ScaleFeature → HandleMV → ReduceDimension×3 | 标签 vocab "unknown"/"1"/"2"；图边去重；节点特征标准化 |
| fraud_detection | ~590k 交易 | `identity`（在 load_data 中已合并） | ~430 → 列过滤后显著缩减 | ExtractDateTimeFeature → TransformPower → RenameColumn → CustomProcess → HashEncode → CustomProcess×2 → LabelEncode → HandleMV | 高维稀疏；高基类别；近 50% 列为空；保留原始类别分布 |
| home_credit | 307 511 行 × 122 列 | bureau, bureau_balance, previous_application, pos_cash, credit_card, installments（共 6 表，pipeline 启用 2 个 JoinTable，3 个被注释） | 122 → 241 | CustomClean×2 → CreateFeature×10 → JoinTable×2 → CustomProcess → LabelEncode → HandleMV → SelectFeature×2 | DAYS_EMPLOYED=365243 哨兵；XNA 字符串；ext_source 衍生 |
| nyc_taxi_hourly | 千万级原始 trip → 小时桶 | 无 | 19 → ~30+ | ParseDate → CustomClean → HandleError(action=delete) → ExtractDateTimeFeature → ResampleTimeSeries → ExtractDateTimeFeature → CreateLagFeature → CreateRollingFeature → CustomProcess → HandleMV → TransformPower → SortRows | parquet；行程时长/距离离群点；resample |
| polish_bankruptcy | year5 | year1, year2, year3, year4（4 个聚合源） | 64 比率 → ~150+ | ConcatTable → TransformPower → CreatePolynomialFeature → JoinTable → CustomProcess×2 → LabelEncode → HandleMV×2 → Oversample | 多年 ARFF；class 列泄漏需删除；64 个金融比率 |

### 2.2 推荐类（RecData）

| 任务 | 交互行数 | user/item 表 | 算子链 | 关键参数 |
|---|---|---|---|---|
| amazon_beauty | All_Beauty.json.gz（数十万评论） | item 元数据（1 张） | JoinTable → FilterKCore → CreateSequence | k=3；max_seq_len=20；leave_one_out；标签由 RecData._apply_label_rule 在 load_data 末尾固化（rating>=4 → 1）|
| movielens | ml-1m（1M 评分） | user, item（2 张） | JoinTable → CustomProcess → FilterKCore → CreateSequence → FilterSample | k=5；max_seq_len=20；leave_one_out；标签 rating>=4 → 1（在 load_data 末尾固化）|
| yelp | Yelp Open Dataset 评论 | user, business（2 张） | JoinTable → CustomProcess → FilterKCore → CreateSequence → FilterSample | k=2；max_seq_len=20；temporal split；标签 stars>=3 → 1（在 load_data 末尾固化）|
| tenrec | QK-video 点击日志 | 无 | FilterSample → FilterKCore | k=3；std-test leave-one-out + 100 fixed negatives；fast 验证读取前 100 万行 |

---

## 3. 复杂度分层

### 高复杂度（★★★★★ 2 个）

- **home_credit / berka**：home_credit 具有 6 张辅助表与 19 步 pipeline；berka 有 8 张原始关系表（loan 主 + 7 辅）、5 次 `ParseDate` 与 4 次 `JoinTable(method="agg")`，是关系结构最复杂的任务之一。
- **home_credit**：行业 benchmark；6 张辅助表 + 122 列原始特征，FeatureEngineer 一次性派生 10 个新特征，处理后膨胀至 241 列，体量与字段语义双重复杂。

### 中高复杂度（★★★★☆ 4 个）

- **citibike_jc_hourly / nyc_taxi_hourly**：单表事件流，但需要哨兵值清洗 + 离群点/错误检测 + ResampleTimeSeries + 滞后/滚动等 **11–13 步**的"脏数据 + 时序"组合。
- **fraud_detection**：宽表（~430 列），通过 CustomProcess / HashEncode 处理高缺失、高基类别和稀疏特征。
- **polish_bankruptcy**：5 张同结构 ARFF 表，4 次 `JoinTable(method="agg")` 聚合跨年特征。
- **elliptic_bitcoin**：唯一的图任务，3 张表 + 自定义 `JoinTable`/`ScaleFeature` 算子。

### 中复杂度（★★★☆☆ 4 个）

- **beijing_air_quality / bike_sharing / bondora / movielens / yelp**：典型时序回归或推荐流水线，算子数 5–13，含 1–2 张辅助表。

### 低复杂度（★★☆☆☆ 及以下 4 个）

- **amazon_beauty / default_credit**：一个是最小推荐流水线（3 步），一个是结构清晰的中等表格流水线（11 步）。
- **tenrec**：仅 1 个预处理算子（FilterKCore），是 dppbench **最简任务**。

---

## 4. 关键指标速览

| 指标 | 最大值 | 任务 | 最小值 | 任务 |
|---|---|---|---|---|
| 辅助表数量 | 7 | berka | 0 | citibike_jc_hourly / nyc_taxi_hourly / tenrec |
| 算子数量 | 19 | home_credit | 1 | tenrec |
| 原始特征列 | ~430 | fraud_detection | ~5–10 | tenrec / amazon_beauty |
| 处理后特征 | 241 | home_credit | ~10 | tenrec |
| 行数级别 | 千万级 | nyc_taxi_hourly（原始 trip） | 数千 | berka loan |

---

## 5. 算子使用频率

> 统计自全部 15 个 task 的 `pre_process.yaml`（截至 papers 算子补齐 D 批次回归后）。
> 全部新增的 paper 类算子（C1/C2/C3）均已在至少一个 task 接入并通过 smoke test。

### 5.1 高频核心算子（≥4 个 task 使用）

| 算子 | 使用任务数 | 主要场景 |
|---|---|---|
| CustomProcess | 12 | 预处理阶段承接列过滤、高基类别频次编码等任务定制逻辑 |
| HandleMV | 11 | 几乎所有 TabularData 流水线收尾 |
| JoinTable | 11 | 跨表多对一聚合（max_cols=20）或 key join |
| LabelEncode | 9 | 类别特征统一编码 |
| TransformPower | 7 | log/sqrt/box-cox/yeo-johnson/quantile 数值变换 |
| CreateFeature | 6 | 特征派生（source_cols + method；内置 mean/sum/std/min/max/median/product/diff/ratio/inc_ratio/concat/identity，或用户自定义 callable） |
| ExtractDateTimeFeature | 6 | hour/day_of_week/month 提取 |
| CustomClean | 5 | 清洗阶段承接哨兵值/异常字符串替换等任务定制逻辑 |
| CreateRollingFeature / CreateLagFeature | 4 | 时序窗口特征 |
| FilterKCore | 4 | RecData k-core 过滤 |

### 5.2 中频领域算子（2–3 个 task）

| 算子 | 使用任务数 | 主要场景 |
|---|---|---|
| CreateSequence / HandleError | 3 | RecData 序列构建；规则约束错误检测 |
| ResampleTimeSeries / HandleError(action=delete) | 2 | 事件流→小时桶 |
| ScaleFeature / ParseDate / SelectFeature / ReduceDimension | 2–3 | 缩放、日期解析、选择与降维 |
| FilterSample | 4 | RecData ID/label 非空过滤 |

### 5.3 低频专用算子（1 个 task，覆盖 paper 算子集）

按 Figure 2 分类：

- **Cleaning**：Deduplicate（elliptic_bitcoin）、HandleError（bondora）、HandleOutlier（citibike_jc_hourly）、HandleMV(method=iterative/knn)（polish_bankruptcy/default_credit）、CustomClean（home_credit / citibike_jc_hourly）
- **Integration**：JoinTable（elliptic_bitcoin / 多表聚合）、ConcatTable（polish_bankruptcy）、SplitColumn（berka）
- **Preprocessing - Encoding**：OneHotEncode（default_credit）、OrdinalEncode（berka）、HashEncode（fraud_detection）
- **Preprocessing - Scaling/Transform**：ScaleFeature（elliptic_bitcoin / beijing_air_quality / bondora）、TransformPower（nyc_taxi_hourly / bike_sharing）
- **Preprocessing - Imbalance**：Oversample（polish_bankruptcy）、Undersample（bondora）
- **Feature Engineering - Generation**：CreatePolynomialFeature（polish_bankruptcy）、CrossFeature（berka）、RenameColumn（fraud_detection）
- **Feature Engineering - Selection**：SelectFeature（home_credit / default_credit）
- **Feature Engineering - Reduction**：ReduceDimension（elliptic_bitcoin / default_credit）
- **Reshape / Sort**：SortRows（nyc_taxi_hourly）

### 5.4 算子原子性边界（避免重复）

为保证每个算子单一职责、可独立组合，下列重叠的语义被**人为拆分**：

| 拆分前 | 拆分后 | 边界 |
|---|---|---|
| `FeatureEngineer` (含 log) | `CreateFeature` + `TransformPower(method="log")` | CreateFeature 做 ratio/sum/std/mean 等 arithmetic；TransformPower 做分布变换 |
| `FilterFeatures`（同时按列名/方差/缺失率筛） | `CustomProcess(mode="drop_columns/drop_high_null")` + `SelectFeature(method="variance")` | 自定义列过滤与统计式特征选择分离 |
| 多个 scaler | `ScaleFeature(method=...)` | 用 method 显式选择 standard/minmax/maxabs/robust/l2 |
| 多个 imputer | `HandleMV(method=...)` | method 覆盖 constant/mean/median/mode/knn/iterative；`action ∈ {delete, impute}` |
| `OneHotEncode` / `OrdinalEncode` / `LabelEncode` / `HashEncode` / `CustomProcess(mode="frequency_encode")` | 全部保留 | 编码语义不同：OHE 增列、Ord 保留有序、Label 单列编码、Hash 固定桶、频次编码走定制预处理 |
| 多个降维算子 | `ReduceDimension(method=...)` | method 覆盖 PCA/SVD/LDA/kernel PCA/UMAP |
| 多个不平衡处理算子 | `Undersample` / `Oversample` | 欠采与过采保留为两个方向，具体算法由 method 选择 |

### 5.5 paper 算子覆盖一览（Figure 2 Taxonomy）

- Integration：JoinTable、ConcatTable、AlignSchema、RenameColumn、CastType、ParseDate、ParseNumber、SortRows、SplitColumn、CustomTransform
- Cleaning：HandleMV、HandleOutlier、HandleError、HandleNonIID、ReweightUPG、CorrectLabel、Deduplicate、CorrectTypo、CustomClean
- Preprocessing：OneHotEncode、OrdinalEncode、LabelEncode、HashEncode、TargetEncode、ScaleFeature、TransformPower、DiscretizeFeature、ClipOutlier、FilterSample、SampleNegative、FilterKCore、Undersample、Oversample、AugmentMixup、AugmentNoise、CustomProcess
- Feature Engineering：CreateFeature、CreatePolynomialFeature、CrossFeature、AggregateGroupFeature、ExtractDateTimeFeature、CreateLagFeature、CreateRollingFeature、ResampleTimeSeries、CreateSequence、TruncateSequence、SelectFeature、ReduceDimension、ExtractTextFeature、ExtractTextEmbedding、ExtractGraphFeature、CustomFE
- Reshape/Sort：SortRows
- RecData：FilterKCore、CreateSequence、FilterSample、SampleNegative（标签由 RecData._apply_label_rule 在 load_data 末尾固化）

---

## 6. 备注

- 所有 TabularData 任务的 `pre_process.yaml` 均以 `LabelEncode` + `HandleMV(method="median")` 收尾，保证下游 LightGBM 直接可训。
- RecData 的 std-test 切分由冻结数据和 `RecData.split()` 处理，`pre_process.yaml` 不再声明 `DataSplit`。
- 仅 `elliptic_bitcoin` 接入 GNN（GCN/GraphSAGE/GAT），需要在 [`scripts/train.py`](file:///Users/bytedance/Documents/dppbech/scripts/train.py) 中走专门的 GNN 分支。
- [`ifasion/`](file:///Users/bytedance/Documents/dppbech/dppbench/tasks/ifasion) 暂未集成；如需上线需补 `ifasion_data.py` + `pre_process.yaml` + `model.yaml`。

---

## 7. 标准测试集说明（Standard Test Set）

为了公平比较各 baseline 自动生成的预处理 pipeline，每个 task 都维护一份**预先冻结、所有 baseline 共用**的标准测试集（std-test）。所有 baseline 跑完后产生的 `best_pipeline.yaml`，都会在该 task 的标准测试集上回放预处理 → 训练 → 评估，得到的 `std_test_<metric>` 才是横向可比的对外指标。

### 7.1 一键生成

```bash
python scripts/build_std_test.py                # 全部 15 个 task
python scripts/build_std_test.py --data_names fraud_detection,movielens
python scripts/build_std_test.py --dry_run      # 只打印切分摘要，不落盘
```

固定参数：`seed = 42`，holdout 比例 `20%`，rec 任务每条 std-test 正例配 `100` 条预先抽样的固定负例。这些值写死在脚本内，不暴露 CLI，避免被无意改动而破坏可比性。

### 7.2 各 task 切分规则

| 任务类型 | 任务名 | 切分方式 | std-test 大小 | 是否含负例采样 |
|---|---|---|---|---|
| Tabular 二分类 | `berka`、`bondora`、`default_credit`、`fraud_detection`、`home_credit`、`polish_bankruptcy` | Stratified holdout（按标签分层）| 训练集 20% | 否 |
| Tabular 时序回归 | `beijing_air_quality`、`bike_sharing`、`citibike_jc_hourly`、`nyc_taxi_hourly` | 按时间列尾部 20% chronological holdout（不打乱）| 训练集尾部 20% | 否 |
| Tabular 图（Graph） | `elliptic_bitcoin` | 按节点 id 随机 20% holdout（保留完整边集） | 有标签节点 20% | 否 |
| Rec | `amazon_beauty`、`movielens`、`yelp`、`tenrec` | Leave-one-out（每用户最近 1 条交互）| 用户数 × 1 | **是**：每条正例配 100 条预抽样固定负例 |

### 7.3 落盘文件结构

```
dppbench/tasks/<task>/std_test/
    std_test.parquet              # 带 label 的 holdout（所有 task 必有）
    train_frozen.parquet          # tabular：去掉 std-test 那 20% 行的剩余训练集
    interaction_frozen.parquet    # rec：去掉 std-test 那条交互的剩余 interaction 表
    std_test_negatives.parquet    # rec：每条 std-test 正例对应的 100 条固定负例
    meta.json                     # {"split_method": ..., "size": ..., "seed": 42, ...}
```

### 7.4 算子是否作用于 std-test 的判定

**完全由算子源码自带的类属性 `APPLIES_TO_STD_TEST` 决定**，与 `pre_process.yaml` 完全解耦：

- 在 [base_op.py](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/base_op.py) 的 `BaseOp` 上默认 `APPLIES_TO_STD_TEST = True`；所有 (A) Fit-on-train+transform-on-both（如 `LabelEncode`、`ScaleFeature`、`HandleMV(action=impute)`）与 (B) 结构性特征构造（如 `JoinTable`、`CreateSequence`、`CustomProcess`）算子默认继承 `True`，与训练数据走同样的 `transform`，保证特征维度对齐。
- (C) 训练专属算子在自己的源码文件里 override 为 `False`：`HandleOutlier`、`HandleError`、`HandleNonIID`、`ReweightUPG`、`Deduplicate`、`FilterSample`、`SampleNegative`、`FilterKCore`、`Undersample`、`Oversample`、`AugmentMixup`、`AugmentNoise`（`HandleMV` 仅在 `action=delete` 下置 `False`，运行时切换）。
- baseline（SAGA / CtxPipe / DiffPrep / SPIO / ReAct / Learn2Clean / DataMaster / DeepPrep / AutoPrep / AlphaClean）和 task 维护者**都不需要感知 / 输出 / 声明**这个字段；他们生成 / 维护的 `pre_process.yaml` 内容保持不变。
- 框架（`run_pre_process`，[dataset.py](file:///Users/bytedance/Documents/dppbech/dppbench/dataset.py)）在拿到 op 实例后读 `op.APPLIES_TO_STD_TEST` 自行决定是否把 `__split__ == "std_test"` 的行临时摘出再拼回。

### 7.5 评测兼容性

- 训练脚本（[scripts/train.py](file:///Users/bytedance/Documents/dppbech/scripts/train.py)、[scripts/train_tabular.py](file:///Users/bytedance/Documents/dppbech/scripts/train_tabular.py)）在打印 val 指标后会额外打印 `std_test: {...}`。
- 基线评测脚本（[scripts/evaluate_tabular_baselines.py](file:///Users/bytedance/Documents/dppbech/scripts/evaluate_tabular_baselines.py)）在终端表格与 CSV 中新增 `std_test_auc`、`std_test_metric` 两列，作为对外公平比较的主指标；rec 任务也走同一套 harness。
- **向下兼容**：若某个 task 目录下没有 `std_test/` 子目录，所有训练 / 评测脚本会自动退化为旧行为（不接入 std-test、`std_test_*` 指标为 `None`）。
