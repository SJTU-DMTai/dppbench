"""Official FlowPipe cached Llama-3.1 semantic-context protocol."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch

from baselines.common.op_spec import OpCategory
from baselines.common.state_features import unique_value_count
from baselines.common.llm_quantization import (fp8_quantization_config, local_torch_dtype,
    require_accelerate, model_dtype_kwargs)


class SemanticContextLoader:
    """Load the official contextual-semantic hidden-state embedding.

    Official FlowPipe pre-generates four embedding styles but its current
    ``generate_context_embedding`` returns only ``contextual_semantic``.
    DPPBench follows that executable behavior exactly.
    """

    def __init__(
        self, cache_dir: str, model_name: str, ctx_dim: int = 4096,
        llama_model_path: str = "/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.model_name = str(model_name)
        self.ctx_dim = int(ctx_dim)
        self.llama_model_path = Path(llama_model_path)

    def path_for(self, data_name: str) -> Path:
        # A generic cache path may hold another model's same-width embedding.
        return self.cache_dir / self.model_name / "contextual_semantic" / f"{data_name}.pt"

    def load(self, data_name: str) -> torch.Tensor:
        path = self.path_for(data_name)
        if not path.is_file():
            raise FileNotFoundError(
                "FlowPipe requires the frozen model-specific contextual-semantic "
                f"embedding for {data_name}: {path}"
            )
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            value = value.get("embedding", value.get("hidden_state"))
        value = torch.as_tensor(value, dtype=torch.float32).squeeze()
        if value.numel() != self.ctx_dim:
            raise ValueError(
                f"FlowPipe context must have {self.ctx_dim} values, got {value.numel()}"
            )
        return value.reshape(self.ctx_dim)

    def resolved_model_path(self) -> Path:
        root = self.llama_model_path
        candidates = (
            root,
            root / self.model_name,
            root / "Meta-Llama-3.1-8B-Instruct",
            root / "Llama-3.1-8B-Instruct",
        )
        for candidate in candidates:
            if (candidate / "config.json").is_file():
                return candidate
        raise FileNotFoundError(
            "Llama checkpoint was not found under llama_model_path="
            f"{root}; expected a model directory containing config.json"
        )

    @staticmethod
    def build_prompt(data_name: str, frame: pd.DataFrame, ctx) -> str:
        """Build the official metadata/statistics-only semantic prompt."""
        rows = max(1, len(frame))
        schema = []
        for column in frame.columns:
            series = frame[column]
            missing = float(series.isna().sum()) / rows
            unique = unique_value_count(series)
            role = "target" if column == ctx.target_col else "feature"
            schema.append(
                f"- {column}: dtype={series.dtype}, role={role}, "
                f"missing={missing:.4f}, unique={unique}"
            )
        categories = ", ".join(category.value for category in OpCategory)
        return (
            "EMBED-ONLY PROMPT -- GLOBAL CONTEXT FOR AUTOML PIPELINES\n"
            "Encode only dataset semantics and operational preparation priors. "
            "Do not generate a pipeline or discuss model training. The static "
            "embedding will condition a FiLM-modulated GFlowNet.\n"
            f"Dataset: {data_name}\nTask type: {ctx.task_type}\n"
            f"Shape: {frame.shape[0]} rows x {frame.shape[1]} columns\n"
            f"Target: {ctx.target_col}\nAvailable component categories: {categories}\n"
            "Summarize missingness, types, categorical cardinality, numeric scale, "
            "skew/outliers, redundancy, useful feature interactions, and operator "
            "applicability for logical-order and primitive selection.\nSCHEMA:\n"
            + "\n".join(schema)
            + "\n[CTX_END]"
        )

    def precompute(
        self, data_name: str, frame: pd.DataFrame, ctx, *, device: str,
        max_length: int = 4096, overwrite: bool = False,
    ) -> Path:
        """Extract the last valid token from the final Llama hidden layer."""
        output_path = self.path_for(data_name)
        if output_path.is_file() and not overwrite:
            return output_path
        model_path = self.resolved_model_path()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_device = torch.device(device)
        require_accelerate(str(torch_device))
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, **model_dtype_kwargs(local_torch_dtype(torch_device)),
            device_map=str(torch_device),
            quantization_config=fp8_quantization_config(torch_device),
        )
        model.eval()
        prompt = self.build_prompt(data_name, frame, ctx)
        tokens = tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True,
            max_length=int(max_length), add_special_tokens=True,
        ).to(torch_device)
        with torch.no_grad():
            result = model(**tokens, output_hidden_states=True, use_cache=False)
        hidden = result.hidden_states[-1]
        last_index = tokens["attention_mask"].sum(dim=1).sub(1).clamp(min=0)
        embedding = hidden[
            torch.arange(hidden.shape[0], device=torch_device), last_index,
        ].detach().cpu().to(torch.float32)
        if embedding.shape != (1, self.ctx_dim):
            raise ValueError(
                f"Llama embedding must have shape (1, {self.ctx_dim}), "
                f"got {tuple(embedding.shape)}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(embedding, temporary)
        os.replace(temporary, output_path)
        return output_path


__all__ = ["SemanticContextLoader"]
