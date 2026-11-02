# 三方法预训练 checkpoint 交付与使用

## 1. 接收方智能体先读

本交付包含 CtxPipe、FlowPipe 的 LightGBM/DIN agent，以及 HAIPipe 的 LightGBM agent 和 scorer。它们是 pipeline 搜索策略，不是最终下游 DIN/LightGBM 模型，也不包含 Qwen3。

使用包含本文档的同一提交中的源码。交付包的 manifest.json 另记录打包时的 Git HEAD 和工作树状态；如果状态非空，该 HEAD 不是交付源码的完整身份，实际应以交付方随后提交本文档和配套代码的提交为准。不要把权重直接交给旧 jiaqinyi 实现加载。

重要约束：

- 不修改原始 checkpoint metadata，不禁用严格校验，不拿 DIN 名称冒充 LightGBM。
- HAIPipe 的推荐任务在非 scratch 模式下暂时自动使用 LightGBM 策略及同目录 scorer，目标下游模型仍是 DIN。输出 provenance 记录 source_model、target_model、policy_model_vocabulary 和 cross_model_transfer。
- 当前交付没有 HAIPipe/DIN 专属预训练权重。即使将来该权重生成，当前代码仍执行上述迁移规则，切换必须明确修改并验证。
- FlowPipe/CtxPipe 的 Qwen3 规范化绝对路径属于 checkpoint 合同。更换路径不保证兼容；只改 pretrain.json 不能改变 .pt 内的合同。
- 运行目录必须与源权重目录分开，不覆盖交付文件。
- 只加载可信来源的 .pt；包内 manifest.json 提供文件大小和 SHA-256，SHA256SUMS 可在传输后校验。

## 2. 交付内容

解压后的目录结构：

```text
pretrained_delivery_20261102/
  manifest.json
  SHA256SUMS
  FlowPipe/
    LightGBM/
      checkpoint.pt
      pretrain.json
      context_cache/Qwen3-8B/contextual_semantic/<task>.pt
    DIN/
      checkpoint.pt
      pretrain.json
      context_cache/Qwen3-8B/contextual_semantic/<task>.pt
  CtxPipe/
    LightGBM/checkpoint.pt
    LightGBM/pretrain.json
    DIN/checkpoint.pt
    DIN/pretrain.json
  HAIPipe/
    LightGBM/aipipe.pt
    LightGBM/alternative_evaluator.pt
    LightGBM/pretrain.json
```

必需与可选文件：

| 文件 | 用途 |
| --- | --- |
| FlowPipe/CtxPipe checkpoint.pt | 默认策略加载入口使用的最终权重 |
| HAIPipe aipipe.pt + alternative_evaluator.pt | 两者必须配套，放在同一目录 |
| pretrain.json | 训练来源/预算/合同和全量启动器的预检信息 |
| FlowPipe context_cache | 已有任务运行需要；缺失不会自动生成 |
| CtxPipe context cache | 本包不带；运行时由 Qwen3 生成 |

不需要交付 frame/combo 中间权重、重复 gfn.pt/q_network.pt、source_logs、训练日志、.venv_gpu。不要将 ckpts 或整个虚拟环境加入普通 Git 历史。

原始来源（GPU 服务器）：

```text
/mnt/cephfs/project_wx-rec-alg-bdc-exp/hachimideng/dppbench/ckpts/pretrained/seed42_v3_20261026_213243/FlowPipe/{LightGBM,DIN}/
/mnt/cephfs/project_wx-rec-alg-bdc-exp/hachimideng/dppbench/ckpts/pretrained/seed42_v3_retry_20261029_211708/CtxPipe/{LightGBM,DIN}/
/mnt/cephfs/project_wx-rec-alg-bdc-exp/hachimideng/dppbench/ckpts/pretrained/seed42_v3_retry_20261029_211708/HAIPipe/LightGBM/
```

本机工具的共享盘挂载为 /mnt/cephfs_zjk/；给 GPU 服务器执行的命令使用 /mnt/cephfs/，不要直接套用工具挂载点。

## 3. 训练设置与兼容合同

交付五组的训练来源均为 seed=42、small_n=1500、fast_train=false、held_out_tasks=[]，属于 seen-task 跨任务预训练，不是未见任务泛化实验。

- LightGBM 源任务：berka、default_credit、fraud_detection、home_credit、polish_bankruptcy、beijing_air_quality、bike_sharing、citibike_jc_hourly、nyc_taxi_hourly。
- DIN 源任务：amazon_beauty、kuairec、movielens、yelp。
- CtxPipe/HAIPipe：56000 frames；FlowPipe：20 epochs。
- reward：bounded_score_or_reciprocal_error_v1。
- search_semantics：primary_schema_random_subset_rec_optional_fe_v2。

加载器还检查算子顺序、action/逻辑布局、状态版本和 encoder；张量维度相同不是完整兼容证据。

本次源码保留三方法及动态选列/随机列采样，但通用 dppbench 数据处理、任务细节与此前比较的 jiaqinyi b093fae 版本一致。部分交付权重是在恢复通用数据代码之前训练的；旧实验分数不能直接视为当前代码的分数。metadata 不覆盖全部数据实现版本，必须重新验证当前实验结果。

## 4. 模型与数据前置条件

CtxPipe 需要完整 Qwen/Qwen3-8B（config、tokenizer、全部权重分片）。FlowPipe 对包内任务使用缓存，可不现场加载编码器，但仍检查其身份路径；新任务预计算缓存需要 Qwen3。HAIPipe 不需要 LLM。

四组 FlowPipe/CtxPipe 源合同中的 Qwen3 路径为：

```text
/mnt/cephfs/project_wx-rec-alg-bdc-exp/hachimideng/dppbench/ckpts/Qwen3-8B
```

新机器使用其他真实路径会报 Incompatible agent checkpoint。普通软链接会被 realpath 展开，不一定解决问题。可保留上述规范化路径；否则先确认完全相同的模型权重，再设计显式迁移转换，同时更新 .pt 合同及 JSON，记录迁移来源。本交付不包含自动迁移工具。

数据不在包内。数据布局为：

```text
<DATA_DIR>/<task>/data/
<DATA_DIR>/<task>/std_test/meta.json
<DATA_DIR>/<task>/std_test/std_test.parquet
<DATA_DIR>/<task>/std_test/train_frozen.parquet        # 表格任务
<DATA_DIR>/<task>/std_test/interaction_frozen.parquet  # 推荐任务
```

复现实验还需匹配的数据版本、冻结划分与任务 YAML。若接收方没有冻结文件，用 scripts/build_std_test.py 显式指定上述任务构建；确认 SUMMARY 全部成功。正式评估加 --skip_build，避免静默重建不同划分。

在目标机器重新安装环境，不搬运 .venv。交付方 GPU 服务器曾使用 Python 3.12、PyTorch 2.8.0 cu128、Transformers 4.56.2，其他依赖见 requirements.txt。requirements.txt 的旧 cu118 源不应覆盖已选的 cu128 torch。上述版本不是完全锁定环境；需运行 pip check 与 CUDA 计算检查。

## 5. 解压与验证

以下命令在 GPU 服务器执行；先下载归档到 ROOT/ckpts，再解压。

```bash
export ROOT=/mnt/cephfs/project_wx-rec-alg-bdc-exp/hachimideng/dppbench
export PYTHON="$ROOT/.venv_gpu/bin/python"
export DELIVERY="$ROOT/ckpts/pretrained_delivery_20261102"
export DATA_DIR="$ROOT/dppbench/tasks"
export QWEN3="$ROOT/ckpts/Qwen3-8B"

cd "$ROOT/ckpts"
tar -xzf pretrained_delivery_20261102.tar.gz
cd "$DELIVERY"
sha256sum -c SHA256SUMS
cd "$ROOT"

"$PYTHON" -m unittest \
  scripts.test_baseline_training \
  baselines.FlowPipe.test_flowpipe_alignment \
  baselines.CtxPipe.test_context \
  baselines.HAIPipe.test_haipipe \
  baselines.SAGA.test_full42_regressions -q
```

2026-11-02 交付方 GPU 服务器已通过上述 37 项单元测试。不能用它替代接收机器上的真实 checkpoint/数据试跑。

## 6. 单任务试跑

先用小型任务核实权重加载、pipeline 和 std-test 输出。以下使用 GPU 0；请选空闲卡。

### CtxPipe / default_credit

```bash
OUT="$ROOT/outputs/handoff_ctx_$(date +%Y%m%d_%H%M%S)"
DPPBENCH_LLM_QUANTIZATION=none "$PYTHON" -u "$ROOT/scripts/evaluate_baseline.py" \
  --baseline CtxPipe --data_names default_credit --model LightGBM \
  --training_mode finetune --pretrained_dir "$DELIVERY" \
  --ctxpipe_gte_model_path "$QWEN3" --finetune_frames 2000 \
  --data_dir "$DATA_DIR" --gpu_id 0 --seed 42 --skip_build \
  --output_dir "$OUT" --output_csv "$OUT/CtxPipe/default_credit_LightGBM/metrics.csv"
```

### FlowPipe / berka

```bash
OUT="$ROOT/outputs/handoff_flow_$(date +%Y%m%d_%H%M%S)"
DPPBENCH_LLM_QUANTIZATION=none "$PYTHON" -u "$ROOT/scripts/evaluate_baseline.py" \
  --baseline FlowPipe --data_names berka --model LightGBM \
  --training_mode finetune --pretrained_dir "$DELIVERY" \
  --qwen3_model_path "$QWEN3" --finetune_epochs 2 \
  --data_dir "$DATA_DIR" --gpu_id 0 --seed 42 --skip_build \
  --output_dir "$OUT" --output_csv "$OUT/FlowPipe/berka_LightGBM/metrics.csv"
```

不要给 FlowPipe 显式传一个空 --context_cache_dir；默认回退到源 checkpoint 同目录 context_cache。新任务须先预计算匹配的训练上下文。

### HAIPipe / movielens（表格策略迁移到推荐）

```bash
OUT="$ROOT/outputs/handoff_hai_$(date +%Y%m%d_%H%M%S)"
"$PYTHON" -u "$ROOT/scripts/evaluate_baseline.py" \
  --baseline HAIPipe --data_names movielens --model DIN \
  --training_mode finetune --pretrained_dir "$DELIVERY" --finetune_frames 2000 \
  --data_dir "$DATA_DIR" --gpu_id 0 --seed 42 --skip_build \
  --output_dir "$OUT" --output_csv "$OUT/HAIPipe/movielens_DIN/metrics.csv"
```

此命令读取 DELIVERY/HAIPipe/LightGBM 的策略及 scorer；实际最终训练 DIN。核查 haipipe_log.json 的 training_provenance，不将结果报告为 DIN 专属源预训练。

finetune 更新目标搜索 agent；pretrained 不更新 RL agent，但 HAIPipe 仍有 scorer adaptation。scratch 不加载本包权重。单任务命令未强制统一各方法搜索采样/快速预算；需要全量搜索可加 --small_n 0 --no-fast_train。

## 7. 一键 39 项全量实验

单任务试跑通过后：

```bash
OUT="$ROOT/outputs/handoff_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT/outputs"
nohup "$PYTHON" -u "$ROOT/scripts/run_full_baseline_experiments.py" \
  --flow_pretrained_dir "$DELIVERY" --pretrained_dir "$DELIVERY" \
  --qwen3_model_path "$QWEN3" --data_dir "$DATA_DIR" \
  --gpus 0,1,2,3,4,5 --training_mode finetune \
  --output_dir "$OUT" > "${OUT}.launcher.log" 2>&1 < /dev/null &
echo "PID=$! OUTPUT=$OUT"
```

默认 3 方法 × 13 task；HAIPipe 推荐组也运行，但使用 LightGBM source。若要排除该组，追加 --skip_haipipe_din 并提供 5 个 GPU ID。

启动器按 baseline/model 分为六组并行，组内顺序执行。全量训练池 small_n=0，搜索 no-fast_train，最终独立 std-test 评估。全量 Yelp 的主机内存及 I/O 开销大，应先试跑再并行。

输出：

```text
<OUT>/<baseline>/<task>_<model>/best_pipeline.yaml
<OUT>/<baseline>/<task>_<model>/metrics.csv
<OUT>/<baseline>/<task>_<model>/status.json
<OUT>/<baseline>/<task>_<model>/launch.log
<OUT>/experiment.json
<OUT>/summary.json
```

看 status.json 和 metrics.csv 的 error、std_test_error、baseline_eval_error，以及正的 std_test_n_rows 和有限 std_test_metric。进程退出 0、候选 fitness 或下游 Epoch 10/10 都不单独证明正式实验成功。summary.json 按组完成后写入，组未结束时也应查看单任务 status。AUC 越高越好，RMSE 越低越好。

## 8. 常见故障处理边界

- Incompatible agent checkpoint：检查源码、模型、搜索合同、Qwen realpath；不删除 metadata 或使用 strict=False 偷过。
- Flow contextual-semantic embedding missing：检查源模型同目录缓存，移除错误的空缓存覆盖，或只用冻结训练池预计算新任务缓存。
- HAIPipe rec 寻找 DIN 权重：接收方可能使用旧 runner；当前默认应寻找 LightGBM 两件套。
- .venv/bin/python 存在却不能执行：检查失效链接，不跨 Python 版本改链接；重新创建目标机环境。
- 不同分支评估结果不一致：核对任务数据实现、冻结划分、YAML、采样预算和奖励预算，不只核对 .pt。
- git 不提交本包：这是预期。Git 交付代码和本文档，归档通过共享盘/制品存储传递；不要提交 Qwen3、任务数据或虚拟环境。
