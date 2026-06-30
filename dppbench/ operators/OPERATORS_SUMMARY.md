# dppbench 算子总览（52 Operators）

本文档对应当前 `/dppbench/ operators/` 下可反射加载的 52 个具体算子，不含 `base_op.py`、`custom_op.py` 和 `__init__.py`。算子按用户更新后的四个阶段组织：S1 Data Integration、S2 Data Cleaning、S3 Data Preprocessing、S4 Feature Engineering。

## S1. Data Integration（11）

| 算子 | 文件 | 功能 |
| --- | --- | --- |
| `JoinTable` | [`join_table.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/integration/join_table.py) | 按 `method=key/agg/rec` 执行键连接、聚合后连接或推荐任务 user/item 侧表连接。 |
| `ConcatTable` | [`concat_table.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/integration/concat_table.py) | 多张同 schema 表纵向拼接，也支持横向拼接。 |
| `AlignSchema` | [`align_schema.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/integration/align_schema.py) | 按字段映射和 dtype 映射对齐 schema，可补 required columns。 |
| `RenameColumn` | [`rename_column.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/integration/rename_column.py) | 显式重命名列。 |
| `DropColumns` | [`drop_columns.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/integration/drop_columns.py) | 按显式列名删除列。 |
| `CastType` | [`cast_type.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/integration/cast_type.py) | 将列转换为 int/float/string/bool/category/datetime 等 dtype。 |
| `ParseDate` | [`parse_date.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/integration/parse_date.py) | 解析字符串日期、YYMMDD 整数日期和 Berka birth_number。 |
| `ParseNumber` | [`parse_number.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/integration/parse_number.py) | 将字符串数值列清洗并解析为 numeric dtype。 |
| `SortRows` | [`sort_rows.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/integration/sort_rows.py) | 按一个或多个键稳定排序。 |
| `SplitColumn` | [`split_column.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/integration/split_column.py) | 按分隔符或正则将复合列拆成多列。 |
| `CustomTransform` | [`custom_transform.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/integration/custom_transform.py) | 集成阶段自定义函数或 sandbox code 转换。 |

## S2. Data Cleaning（9）

| 算子 | 文件 | 功能 |
| --- | --- | --- |
| `HandleMV` | [`handle_m_v.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/cleaning/handle_m_v.py) | 缺失值处理：`action ∈ {delete, impute}`；impute 支持 median/mean/mode/constant/knn/iterative。 |
| `HandleOutlier` | [`handle_outlier.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/cleaning/handle_outlier.py) | 异常值处理：检测 + `action ∈ {delete, repair}`，repair 支持 clip/median/set_missing/winsorize。 |
| `HandleError` | [`handle_error.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/cleaning/handle_error.py) | 规则违反错误处理：检测 + `action ∈ {delete, repair}`，repair 支持 set_missing/fill_constant/clip/median/mode。 |
| `HandleNonIID` | [`handle_non_i_i_d.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/cleaning/handle_non_i_i_d.py) | non-IID 样本处理：检测 + `action ∈ {delete, reweight}`，reweight 写 `sample_weight` 列。 |
| `ReweightUPG` | [`reweight_u_p_g.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/cleaning/reweight_u_p_g.py) | 低表现子群直接 up-weight：写 `sample_weight` 列。 |
| `CorrectLabel` | [`correct_label.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/cleaning/correct_label.py) | 基于高置信预测概率翻转或标记疑似错误标签。 |
| `Deduplicate` | [`deduplicate.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/cleaning/deduplicate.py) | 按 subset 和 keep 策略去重。 |
| `CorrectTypo` | [`correct_typo.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/cleaning/correct_typo.py) | 显式 mapping 或 rapidfuzz 可选依赖进行文本纠错。 |
| `CustomClean` | [`custom_clean.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/cleaning/custom_clean.py) | 清洗阶段自定义逻辑，并承接旧 MapValues/ReplaceText 语义。 |

## S3. Data Preprocessing（17）

| 算子 | 文件 | 功能 |
| --- | --- | --- |
| `OneHotEncode` | [`one_hot_encode.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/one_hot_encode.py) | 类别列 one-hot 展开。 |
| `OrdinalEncode` | [`ordinal_encode.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/ordinal_encode.py) | 按显式顺序做有序类别整数编码。 |
| `LabelEncode` | [`label_encode.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/label_encode.py) | 类别值映射为整数。 |
| `HashEncode` | [`hash_encode.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/hash_encode.py) | 高基数类别哈希桶编码。 |
| `TargetEncode` | [`target_encode.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/target_encode.py) | 高基数类别目标均值编码。 |
| `ScaleFeature` | [`scale_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/scale_feature.py) | standard/minmax/maxabs/robust/L2 数值缩放。 |
| `TransformPower` | [`transform_power.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/transform_power.py) | log/sqrt/box-cox/yeo-johnson/quantile 分布变换。 |
| `DiscretizeFeature` | [`discretize_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/discretize_feature.py) | 连续列分箱，支持 manual/uniform/quantile/kmeans。 |
| `ClipOutlier` | [`clip_outlier.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/clip_outlier.py) | 按分位或显式边界 winsorize 截断极值。 |
| `FilterSample` | [`filter_sample.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/filter_sample.py) | 按 query、row func 或 NA subset 过滤样本。 |
| `SampleNegative` | [`sample_negative.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/sample_negative.py) | 推荐任务为正样本采样负例。 |
| `FilterKCore` | [`filter_k_core.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/filter_k_core.py) | 推荐任务 k-core 用户/物品过滤。 |
| `Undersample` | [`undersample.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/undersample.py) | random/tomek/enn 欠采样。 |
| `Oversample` | [`oversample.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/oversample.py) | random/smote/adasyn/smote_nc 过采样。 |
| `AugmentMixup` | [`augment_mixup.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/augment_mixup.py) | Mixup 数据增强。 |
| `AugmentNoise` | [`augment_noise.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/augment_noise.py) | 数值列加噪生成增强样本。 |
| `CustomProcess` | [`custom_process.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/preprocessing/custom_process.py) | 预处理阶段自定义逻辑，并承接高缺失列过滤与频次编码。 |

## S4. Feature Engineering（15）

| 算子 | 文件 | 功能 |
| --- | --- | --- |
| `CreateFeature` | [`create_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/create_feature.py) | 基于算术、比例、聚合或 callable 创建单个新特征。 |
| `CreatePolynomialFeature` | [`create_polynomial_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/create_polynomial_feature.py) | 多项式和交互特征。 |
| `CrossFeature` | [`cross_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/cross_feature.py) | 类别列交叉特征。 |
| `AggregateGroupFeature` | [`aggregate_group_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/aggregate_group_feature.py) | 分组统计特征并回填到原表。 |
| `ExtractDateTimeFeature` | [`extract_date_time_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/extract_date_time_feature.py) | 从 datetime 列提取年/月/星期/小时等。 |
| `CreateLagFeature` | [`create_lag_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/create_lag_feature.py) | 时序滞后特征。 |
| `CreateRollingFeature` | [`create_rolling_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/create_rolling_feature.py) | 时序滚动窗口统计。 |
| `CreateSequence` | [`create_sequence.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/create_sequence.py) | 构造用户历史行为序列。 |
| `TruncateSequence` | [`truncate_sequence.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/truncate_sequence.py) | 截断 list/sequence 特征到固定长度。 |
| `SelectFeature` | [`select_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/select_feature.py) | variance/univariate/rfe/model 特征选择。 |
| `ReduceDimension` | [`reduce_dimension.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/reduce_dimension.py) | PCA/SVD/KernelPCA/LDA/UMAP 降维。 |
| `ExtractTextFeature` | [`extract_text_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/extract_text_feature.py) | TF-IDF/BoW/n-gram 文本统计特征。 |
| `ExtractTextEmbedding` | [`extract_text_embedding.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/extract_text_embedding.py) | 文本 embedding 特征，默认使用确定性 hash embedding。 |
| `ExtractGraphFeature` | [`extract_graph_feature.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/extract_graph_feature.py) | 图/边表节点度、PageRank 等结构特征。 |
| `CustomFE` | [`custom_f_e.py`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/feature_engineering/custom_f_e.py) | 通过 Python callable 执行自定义特征工程。 |

## 原子性与训练/测试约束

- `FIT_ON_TRAIN_ONLY=True` 的算子会在训练 slice 上拟合统计量，再复用到标准测试 slice，避免泄漏；主要包括编码、缩放、填充、分箱、选择和降维。
- `APPLIES_TO_STD_TEST=False` 的算子只改变训练行，标准测试行保持不被采样/删除/增强影响；主要包括 `HandleOutlier`、`HandleError`、`HandleNonIID`、`ReweightUPG`、`Deduplicate`、`FilterSample`、`SampleNegative`、`FilterKCore`、`Undersample`、`Oversample`、`AugmentMixup`、`AugmentNoise`。
- 表外旧工具算子已移除，其语义分别由 `CustomClean`、`CustomProcess` 或对应合并算子的 `method/mode` 承接。
