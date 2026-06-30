# Auto-Prep Baseline

基于 SIGMOD 2024 论文 *Auto-Prep: Holistic Prediction of Data Preparation Steps for
Self-Service BI* 的工程实现，作为 `dppbench` 的一个基线。

## 1. 原理摘要

Auto-Prep 把 BI 项目里"为多张表自动选择数据准备步骤 + 表间连接"建模为
**Multi-table Preparation-By-Probability (MPBP)**：

- 输入：一组表 `T = {T₁, …, Tₙ}` 与操作库 `O`。
- 输出：每张表 `Tᵢ` 上的转换序列 `Sᵢ = (Oᵢ₁, Oᵢ₂, …)`，以及把所有变换后表
  连成一棵 spanning tree 的 `(n-1)` 条 join 边。
- 目标函数：

  ```
  argmax_{S, J(S(T))}  ∏ᵢⱼ p(Oᵢⱼ)  ·  ∏ p(joinᵢⱼ)
  ```

论文用两个概率模型给出 `p(O|T)` 和 `p̃(T, T')`：

- **`M_T` / `M_T+`**：单表/全局特征的 boosted-tree 概率模型，输出 calibrated
  `p(O|T) ∈ [0, 1]`。`M_T+` 还引入 column-header / value-domain 等跨表特征。
- **`M_J`**：连接概率，做 `p(T, T') = max(p̃(T, T'), 0.5)` 归一化。

在线阶段 Auto-Prep 给每张表建 transformation tree（深度 m=2），每条边权
等于 `p(O|T)`；再用 join-edges 把不同子表的叶子相连，得到 global search
graph `G(T)`。MPBP 在 `G(T)` 上的求解由动态规划+最大权重 spanning tree
近似完成。

## 2. 工程映射（dppbench）

| 论文概念 | 本仓库映射 |
|---|---|
| 多张表 `T` | tabular: `train_df / test_df / aux_dfs`；rec: `interaction / user / item` |
| 操作库 `O` | [`dppbench/ operators/`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/) 下全部算子（[`operator_catalog.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/operator_catalog.py) 加载时与共享 catalog 数量保持一致） |
| `M_T+` 概率 `p(O|T)` | 启发式特征先验（缺失率、numeric/categorical 计数、time_col 是否存在、是否倾斜等） + 下游 AUC 在线 multiplicative-weights 更新 |
| `M_J` 连接概率 | schema-driven：rec 主表 join 所有侧表（`JoinTable` 强制）；tabular 每个 aux_df 候选 `JoinTable` / `JoinTable`，初始 `p=0.5` |
| Transformation tree | 每张子表按 table kind（`interaction / user_df / item_df / main_tabular / aux_df`）取专用算子白名单，以 beam search（branching K, depth m=2）展开 |
| MPBP 求解 | 笛卡尔积合成"主表路径 + 侧表路径 + join 步骤" → 按 `∏p(O)·∏p(join)` 排序取 top-N candidate |
| 下游反馈 | 每个 candidate 用 `AutoPrepEvaluator`（`CtxPipeEvaluator` + `small_n` 子采样）跑一次下游训练；mean-centred AUC reward 反向更新 `M_T+` 与 `M_J` |

## 3. 算子库覆盖

[`operator_catalog.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/operator_catalog.py)
按 [`OPERATORS_SUMMARY.md`](file:///Users/bytedance/Documents/dppbech/dppbench/%20operators/OPERATORS_SUMMARY.md)
的章节录入了全部 **59 个算子**：

| # | 章节 | 数量 |
|---|---|---|
| 1 | Cleaning | 11 |
| 2 | Integration | 5 |
| 3 | Preprocessing-Schema | 5 |
| 4 | Preprocessing-Scaling | 6 |
| 5 | Preprocessing-Encoding | 7 |
| 6 | Preprocessing-Imbalance | 5 |
| 7 | FE-Generation | 5 |
| 8 | FE-TimeSeries | 3 |
| 9 | FE-Selection | 2 |
| 10 | FE-Reduction | 4 |
| 11 | Sort | 1 |
| 12 | Recommendation | 5 |
| **合计** |  | **59** |

每条 `OpSpec` 记录 `category / task_type / default_params / valid_targets /
mandatory / prior_features`，其中 `prior_features` 由
[`transformation_model.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/transformation_model.py)
的 `_ctx_signals` 与 sigmoid 转换为冷启动概率，再由后续轮次的下游 AUC 反馈
做 multiplicative-weights 更新。

[`pipeline_factory.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/pipeline_factory.py)
为这 59 个算子分别提供 context-aware 的 `build_default_params`，遇到不可用
的上下文（例如 `ReduceDimension` 无 `target_col`、时序算子无
`time_col`）会返回 `None`，让上层 solver 安全跳过。

## 4. 模块布局

| 文件 | 作用 |
|---|---|
| [`__init__.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/__init__.py) | 暴露 `AutoPrep` 类 |
| [`operator_catalog.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/operator_catalog.py) | 59 个算子的本地 catalog（`OpCategory` 枚举 + `OpSpec` + 启发式特征） |
| [`pipeline_factory.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/pipeline_factory.py) | 59 算子的 context-aware 默认参数构造，输出 `PipelineStep` |
| [`transformation_model.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/transformation_model.py) | `TransformationModel`（M_T+ 启发式 + 在线更新）+ `JoinModel`（M_J） |
| [`transformation_tree.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/transformation_tree.py) | 多表 transformation tree（depth=2 beam search） |
| [`solver.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/solver.py) | MPBP 求解：主表树 × 侧表树 × join 边 → top-N candidate；本地 `repair_pipeline` 知道全部 59 算子 |
| [`evaluator.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/evaluator.py) | `AutoPrepEvaluator(CtxPipeEvaluator)`（小数据下游训练 + `evaluate_for_agent`） |
| [`auto_prep.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/auto_prep.py) | 顶层：搜索 + 下游反馈 + 全量复评 + 日志 |
| [`run_auto_prep.py`](file:///Users/bytedance/Documents/dppbech/baselines/AutoPrep/run_auto_prep.py) | CLI |

## 5. CLI 用法

```bash
# Tabular smoke test
python -m baselines.AutoPrep.run_auto_prep \
    --data_name fraud_detection \
    --n_iters 2 --beam 2 --n_candidates 3 --small_n 1500

# Recommendation smoke test
python -m baselines.AutoPrep.run_auto_prep \
    --data_name movielens \
    --n_iters 2 --beam 2 --n_candidates 3 --small_n 1500

# 完整运行（默认 5 轮，beam=4，n_candidates=8）
python -m baselines.AutoPrep.run_auto_prep --data_name home_credit
```

支持的 `--data_name`：`home_credit / fraud_detection / amazon_beauty /
movielens / yelp / tenrec`。其它参数：

| Flag | 默认 | 说明 |
|---|---|---|
| `--n_iters` | 5 | 外层搜索轮数 |
| `--beam` | 4 | transformation tree 每层 beam 宽度 |
| `--n_candidates` | 8 | 每轮提交评估的 candidate pipeline 数量 |
| `--max_depth` | 2 | transformation tree 深度 m（论文 m=2） |
| `--small_n` | 3000 | 评估子采样量；`0` 表示用全量数据 |
| `--no_eval_full` | off | 跳过最终全量复评 |
| `--eta` | 0.5 | multiplicative-weights 学习率 |
| `--seed` | 42 | 随机种子 |
| `--output_dir` | 自动 | 日志输出目录 |
| `--quiet` | off | 关闭 verbose 日志 |

输出文件：

- `best_pipeline.yaml`：胜出 pipeline 的可读 YAML 序列化。
- `auto_prep_log.json`：每轮 candidate ops、AUC、概率快照，`op_probs`
  字段恒含 59 个 op key，便于审计 multiplicative-weights 是否生效。

## 6. 与论文的取舍

| 维度 | 论文 | 本实现 |
|---|---|---|
| `M_T+` | offline boosted decision trees + global features | 启发式先验（声明在 `OpSpec.prior_features`）+ 在线 multiplicative-weights 更新（用下游 AUC 作 reward） |
| `M_J` | 概率模型估计任意两表 join 概率 | schema-driven：rec 用 `JoinTable`（mandatory，p=1.0）；tabular 每个 aux_df 给 `JoinTable`/`JoinTable` 候选，初始 p=0.5，按下游 AUC 更新 |
| 全局图搜索 | DP + Maximum Weight Spanning Tree | 每张子表 beam search 选 top-K 路径；主-侧 join 直接按概率倒序贪心成树（项目主表只有一棵主树，相当于退化的 MWST） |
| 下游反馈 | 不在论文范围 | 加入：每个 candidate 都跑一次下游训练（LightGBM AUC for tabular / DIN AUC for rec），mean-centred reward 更新 M_T+/M_J |

这些取舍换来：
1. 不需要离线训练 boosted-tree，可以直接跑 dppbench 任意 task；
2. 对 dppbench 已有的 schema-driven join 语义保持兼容；
3. 通过下游反馈把"启发式先验"逐步纠偏到真实任务的最优算子分布。

## 7. 验证

```bash
# 1. catalog 完整性
python -c "from baselines.AutoPrep.operator_catalog import CATALOG; \
  assert len(CATALOG)==58, len(CATALOG); print('58 ops OK')"

# 2. 静态 import
python -c "from baselines.AutoPrep.auto_prep import AutoPrep; \
  from baselines.AutoPrep.solver import solve; \
  from baselines.AutoPrep.transformation_model import TransformationModel, JoinModel; \
  from baselines.AutoPrep.transformation_tree import build_transformation_tree; \
  print('IMPORT OK')"

# 3. CLI help
python -m baselines.AutoPrep.run_auto_prep --help
```
