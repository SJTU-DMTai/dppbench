# LLMAutoDP

本实现把 LLM-AutoDP 的分组策略搜索迁移到 DPPBench。实验表中请使用
**LLMAutoDP**，不要标注为官方完整复现。

参考：[论文 §3.2](https://www.vldb.org/pvldb/vol19/p794-cheng.pdf)、
[官方源码](https://github.com/secretflow/ACoLab/tree/main/Autodp-paper-code)、
[meta_agent.py](https://github.com/secretflow/ACoLab/blob/main/Autodp-paper-code/agentscope/meta_agent.py)。
本目录为独立实现，复用本仓库的公共组件，未拷贝上游代码。

## 原生支持与适配范围

原论文优化问答/对话文本的清洗、优化、生成、选择策略，以 LLM 微调后的效果作为反馈。
本项目 14 个任务（以 scripts/run_task_baselines.py 的 TASKS 为准）均不属于其原生任务。
不能仅把 CSV 转成 JSON 就声称原方法支持 DIN、LightGBM 或 GraphSAGE。

| 数据集 | 当前任务 | 默认下游模型 / 主指标 | 本适配版接入方式 |
|---|---|---|---|
| berka | 信贷二分类 | LightGBM / AUC | 表格及多表准备 |
| default_credit | 违约二分类 | LightGBM / AUC | 表格准备 |
| fraud_detection | 欺诈二分类 | LightGBM / AUC | 表格及多表准备 |
| home_credit | 信贷二分类 | LightGBM / AUC | 表格及多表准备 |
| polish_bankruptcy | 破产二分类 | LightGBM / AUC | 表格准备 |
| beijing_air_quality | 空气质量时序回归 | LightGBM / RMSE | 共享回归评估及时间划分 |
| bike_sharing | 租赁量时序回归 | LightGBM / RMSE | 共享回归评估及时间划分 |
| citibike_jc_hourly | 小时需求预测 | LightGBM / RMSE | 共享回归评估及时间划分 |
| nyc_taxi_hourly | 小时需求预测 | LightGBM / RMSE | 共享回归评估及时间划分 |
| amazon_beauty | 序列推荐 | DIN / AUC | 交互表、用户/商品侧表及历史序列 |
| kuairec | 序列推荐 | DIN / AUC | 同上 |
| movielens | 序列推荐 | DIN / AUC | 同上 |
| yelp | 序列推荐 | DIN / AUC | 同上 |
| elliptic_bitcoin | 图节点二分类 | GraphSAGE / AUC | 节点属性准备，保持共享图训练器及边关系 |

这里的“接入”表示实现与任务接口、模型配置兼容；不代表已经完成这 14 个数据集的真实实验。
图任务没有新增拓扑搜索算子，能力范围受共享算子目录约束。
README 根目录还提及 tenrec，但它不在上述 14 个任务清单内。

## 实现与原方法的差异

* 保留：每轮生成多条不同策略、执行下游训练、将整组 fitness 和相对固定起点的
  `fitness - anchor_fitness` 回传给 LLM，迭代后选取已经验证过的最佳策略。
  默认 6 轮，每轮最多 4 条，允许提前停止；失败候选不会胜出。
* 起点：从空 Pipeline 自动补全基准要求的必要操作（如编码、缺失处理、连接或序列构造）。
  不读取任务 prepare.yaml。此起点是“无可选操作”，并非原论文的完全未处理文本。
* 搜索空间：替换为 DPPBench 的共享任务算子；不再是原论文四类文本团队的 65 种组合。
  保留顺序搜索，允许共享 DSL 的 DAG 表示；候选必须完整合法，不事后修改胜者。
* 反馈：共享验证集分数；RMSE/MAE/MSE/logloss 由公共 evaluator 转负值。
  不向 LLM 反馈 std-test；最终 std-test 交给 evaluate_baseline.py。
* 缓存：同一次搜索中，相同完整 Pipeline 的分数复用。**没有**实现原论文的处理结果
  最长前缀缓存，不能把这项简化声称为完整 CRM。
* DPS / PTS / LESS：未启用。没有使用医学文本质量分类器替代结构化数据质量判断。
  默认 full-data、正常训练；`fast_train: true` 是本基准快速训练选项，不是论文 DPS。
* 使用公共预算（评估次数、LLM 次数、长度和资源限制）及现有 API/本地模型客户端。
  Prompt 只含 schema、算子和验证 fitness，不包含原始样本；schema 仍包含列名。

若必须严格复现原方法，应另设医学问答 SFT 赛道，按官方环境部署训练、推理、LESS
及质量筛选模型。这会改变任务、模型、指标，不能与当前 14 个任务的结果直接混合比较。
若继续拓展适配版：表格需训练独立质量筛选器；时序采样需保持时间块、窗口和因果顺序；
推荐需按用户保留完整历史；图需保持子图/边及 ID 一致。上述采样和质量模型都应仅用训练数据拟合，
作为新增适配机制单独做消融。不要生成/改写真实标签来模拟原论文文本生成。

## 使用

依赖沿用仓库 requirements.txt；不需要安装上游 ms-swift/AgentScope/LESS。
编辑本目录 config.yaml：指定服务支持的 `llm_model`。API 模式沿用根目录 apikeys.json
（也可通过 DPPBENCH_API_KEYS_PATH 指定文件），格式：

```json
{"ProviderName": {"OPENAI_API_KEY": "your-key", "OPENAI_BASE_URL": "https://your-endpoint/v1"}}
```

多个服务时设置 DPPBENCH_LLM_PROVIDER。模型可以通过 config.yaml 的 llm_model
或 LLM_MODEL 环境变量指定；默认客户端模型不一定被你的服务支持。
本地模型设置 `llm_backend: local`、`llm_model: /path/to/model`。
评测入口通过 `--qwen3_model_path` 传入本地模型路径（shell 启动器读取 `QWEN3_MODEL_PATH`），覆盖 YAML 的 backend/model；直接构造类时使用 YAML 配置。

单任务命令（数据目录应包含 `<task>/data`）：

```bash
python scripts/evaluate_baseline.py --baseline LLMAutoDP --data_names default_credit --model LightGBM --data_dir /path/to/datasets --output_dir outputs/autodp --max_evaluations 25 --max_llm_calls 6 --seed 42
```

全部 14 个任务，分别沿用任务默认模型：

```bash
python scripts/evaluate_baseline.py --baseline LLMAutoDP --data_names amazon_beauty,kuairec,movielens,yelp,elliptic_bitcoin,beijing_air_quality,berka,bike_sharing,citibike_jc_hourly,default_credit,fraud_detection,home_credit,nyc_taxi_hourly,polish_bankruptcy --data_dir /path/to/datasets --output_dir outputs/autodp --max_evaluations 25 --max_llm_calls 6 --seed 42
```

25 次是 1 个起点加 6×4 个候选的上限；重复和无效候选可能减少实际训练次数。
最终评估按公共 harness 协议执行。和其他方法比较时统一模型、数据划分、LLM、预算及种子。
结果位于 `outputs/autodp/LLMAutoDP/<task>_<model>/`：best_pipeline.yaml、
trajectory.json（分组分数和错误）、agent_log.json（对话）、run_summary.json（适配标记）。

测试：`python -m pytest baselines/LLMAutoDP/test_llm_autodp.py -q`。
包含三类 DSL 起点合法性、14 个任务配置、回归指标方向、分组反馈、去重、错误处理、预算停止及产物测试；
搜索逻辑测试使用模拟 LLM/训练分数，不需要 API key，不构成真实模型效果验证。

2026-10-15 验证记录：新增测试及 scripts/test_experiment_regressions.py 合计
34 passed、2 subtests passed。本机默认 Python 的 NumPy/SciPy 不兼容，测试使用仓库已有
analysis/test_dependencies 作为 PYTHONPATH，没有修改全局环境。另以本地真实 Bike Sharing
数据、快速 LightGBM 和模拟停止响应验证了训练与产物链路（验证 RMSE 125.3485），
见 outputs/llm_autodp_smoke/bike_sharing/SMOKE_ONLY.md；该数值不是正式 LLM 搜索结果。
尚未执行真实 LLM 调用及全部 14 个数据集的完整实验。

命令中的正式名称为 `LLMAutoDP`；旧名称 `LLM-AutoDP-Adapted` 仍作为兼容别名。两者的新运行结果均写入 `LLMAutoDP` 目录；已有结果目录不会重命名。


## 运行配置对齐（2026-11-13）

与 BAT / DeepPrep 对齐：max_tokens=2048、timeout=600、enable_thinking=false、bfloat16、device_map=auto，沿用共享量化策略。搜索下游评分使用 downstream_eval_n=10000、fast_train=true。

保留方法自己的 temperature=0.6、top_p=1.0、repetition_penalty=1.0；BAT 对应为 0.4/0.8/1.0，DeepPrep 为 0.01/1.0/1.1。保留 n_rounds=6、n_candidates=4，公共入口统一控制长度和调用预算。DeepPrep 使用其专用 DeepPrep-Qwen3-8B 权重，LLMAutoDP 与 BAT 使用基础 Qwen3-8B。

保留 eval_full=false，最终全量评估由公共入口独立执行。直接调用并开启 eval_full 时会取消搜索采样上限并关闭快速训练后再评估。

2048 token 上限也适用于一次生成多个完整候选的 LLMAutoDP，长候选可能截断，需观察实验日志；未为此改变候选数量。
