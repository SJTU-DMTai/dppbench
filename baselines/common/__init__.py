"""Shared baseline runtime utilities.

This package exposes the common DPPBench baseline contract: task loading,
operator metadata, pipeline/DAG helpers, legality checks, and downstream
pipeline evaluation. Baseline-specific search strategies should depend on
these interfaces instead of keeping private operator spaces or executors.
"""
from .data_loader import DATASET_REGISTRY, TaskDataLoader
from .budget import Budget, BudgetExceededError, budget_scope, get_active_budget
from .op_spec import OpCategory, OpSpec
from .utils import get_avilable_ops, get_available_ops, get_data, get_data_ctx, get_op
from .executor import TrainingExecutor
from .data_ctx import DataContext, TableContext, infer_data_context
from .pipeline import (
    Evaluator,
    Pipeline,
    PipelineCheckReport,
    PipelineStep,
    assign_dag_structure,
    build_default_params,
    check_pipeline,
    ensure_tabular_tail,
    finalize_pipeline,
    get_category_order,
    get_order_constraints,
    get_param_space,
    get_source_tables,
    init_piepeline,
    init_pipeline,
    is_legal,
    make_step,
    repair,
    validate_step_params,
    get_operator_param_names,
)
from .evaluator import EvaluationResult, PipelineEvaluator
from .search_evaluator import SearchPipelineEvaluator, subsample_data_inplace
from .state_features import dataframe_state_features, primary_feature_frame
from .op_registry import (
    get_operator_descriptions,
    get_operator_summary,
    get_operator_detail,
)
