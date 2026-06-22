# DPPBench: A Benchmark of Automatic Training Data Preparation Pipeline Orchestration of ML Applications

## Introduction
DPPBench is a benchmark for studying automated training-data preparation in real-world machine learning workflows. In practical ML applications, raw data rarely feeds directly into a model. It must first pass through a complex preparation pipeline composed of data integration, cleaning, preprocessing, and feature engineering operators.

Designing such a pipeline is a time-consuming expert task. Developers must choose which operators to apply, how to order them, and how to tune their hyperparameters for each dataset, task, and downstream model. Recent research has proposed many automated pipeline orchestration methods, including classical search, differentiable pipeline optimization, reinforcement learning, and LLM-driven agents.

DPPBench provides a unified benchmark for evaluating these methods under a shared operator space, task suite, evaluation protocol, and legality constraints. The goal is to make different approaches comparable while preserving the complexity of realistic ML data preparation.

## Pipeline Optimization
DPPBench formulates data preparation as a constrained pipeline optimization problem. Given an operator space, a search method must construct a legal pipeline that maximizes downstream task performance:

```text
P* = argmax_P J(P), subject to legal(P) = 1
```

Here, `P` is a sequence or DAG of data preparation operators, `J(P)` measures the quality of the resulting ML workflow, and `legal(P)` encodes execution validity, structural constraints, and search-budget limits.

## Benchmark Scope
DPPBench aligns prior methods along the dimensions that matter for fair comparison:

- **Operator space**: a standardized set of common operators across integration, cleaning, preprocessing, transformation, and feature engineering.
- **Optimization target**: downstream performance on complex ML tasks rather than isolated data-quality proxies.
- **Constraints**: shared legality checks, execution failure handling, and search-budget control.
- **Task coverage**: recommendation, tabular prediction, time-series prediction, and graph learning tasks that require non-trivial data preparation.

## Installation

Create a Python environment and install the project dependencies:

```bash
conda create -n dppbench python=3.9
conda activate dppbench
pip install -r requirements.txt
```

The default PyTorch wheel index in `requirements.txt` targets CUDA 11.8. If your machine uses a different CUDA or CPU-only setup, install the matching PyTorch build first, then install the remaining requirements.

## Tasks

DPPBench currently includes tabular, time-series, graph, and recommendation tasks under `dppbench/tasks/`. Each task directory provides:

- `pre_process.yaml`: the default DSL preparation pipeline.
- `model.yaml`: model choices, feature configuration, and training parameters.
- `<task>_data.py`: the dataset loader and task-specific data preparation logic.
- `std_test/`: the frozen standard test split when it has been generated.

Supported formal tasks include:

```text
amazon_beauty, beijing_air_quality, berka, bike_sharing, bondora,
citibike_jc_hourly, default_credit, elliptic_bitcoin, fraud_detection,
home_credit, kuairec, movielens, nyc_taxi_hourly, polish_bankruptcy, yelp
```

## Common Commands

Build or refresh frozen standard test data:

```bash
python scripts/build_std_test.py --data_name amazon_beauty --data_dir /path/to/data --gpu_id 0
```

Train and evaluate one task/model with its default DSL preparation pipeline:

```bash
python scripts/train_task_model.py --data_name amazon_beauty --model DIN --data_dir /path/to/data --gpu_id 0
```

Evaluate one baseline on a task:

```bash
python scripts/evaluate_baseline.py --baseline SAGA --data_name amazon_beauty --data_dir /path/to/data --out_dir /path/to/output --gpu_id 0
```

