"""Unified LLM client used by DeepPrep.

Two backends are supported:
  * ``backend="api"``  — OpenAI-compatible HTTP API (default; requires only
    the ``openai`` SDK). API credentials are loaded from the project-level
    ``apikeys.json`` file.
  * ``backend="local"``  — a local Hugging Face ``transformers`` model. This
    is loaded **lazily** inside ``chat()`` so that environments that only have
    the OpenAI SDK installed are unaffected.

The class purposely keeps the surface tiny (``chat(messages, system=None)``)
so that both inference and the optional RL trainer can share it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from baselines.common.api_keys import load_openai_compatible_api_config
from baselines.common.budget import get_active_budget
from baselines.common.llm_quantization import (fp8_quantization_config, local_torch_dtype,
    require_accelerate, model_dtype_kwargs, LocalModelInitializationError)


@dataclass
class LLMConfig:
    backend: str = "api"           # "api" | "local"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.01
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout: int = 600
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    enable_thinking: Optional[bool] = None
    repetition_penalty: float = 1.0


class LLMClient:
    """Tiny wrapper providing a uniform ``chat(messages, system=None)`` API.

    The backend is chosen at construction time. The local backend is loaded
    lazily so the OpenAI-only path has zero extra dependencies.
    """

    def __init__(
        self,
        backend: str = "api",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.01,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        timeout: int = 600,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        enable_thinking: Optional[bool] = None,
        repetition_penalty: float = 1.0,
    ) -> None:
        if backend not in ("api", "local"):
            raise ValueError(f"Unknown LLM backend {backend!r}")
        if api_key is not None or base_url is not None:
            raise ValueError(
                "LLM API credentials must be loaded from apikeys.json; "
                "do not pass api_key/base_url through code or CLI."
            )
        api_config = (
            load_openai_compatible_api_config() if backend == "api" else {}
        )
        resolved_model = model or os.environ.get("DEEPPREP_MODEL")
        if backend == "local" and not resolved_model:
            raise ValueError(
                "backend='local' requires llm_model or DEEPPREP_MODEL to point "
                "to the official DeepPrep checkpoint."
            )
        self.cfg = LLMConfig(
            backend=backend,
            model=resolved_model or os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            api_key=api_config.get("api_key"),
            base_url=api_config.get("base_url"),
            temperature=temperature,
            top_p=float(top_p),
            max_tokens=max_tokens,
            timeout=timeout,
            torch_dtype=torch_dtype,
            device_map=device_map,
            enable_thinking=enable_thinking,
            repetition_penalty=repetition_penalty,
        )
        # Lazy-initialised handles
        self._api_client = None     # openai.OpenAI
        self._local_model = None    # transformers model
        self._local_tokenizer = None
        self.total_tokens_used = 0
        self.last_response: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self.cfg.backend

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a list of ``{"role", "content"}`` messages and return the
        assistant text. ``system`` is prepended to ``messages`` if provided.
        """
        budget = get_active_budget()
        if budget is not None:
            budget.consume_llm_call()
        if system is not None:
            messages = [{"role": "system", "content": system}, *messages]
        if self.cfg.backend == "api":
            answer = self._chat_api(messages, temperature=temperature, max_tokens=max_tokens)
        else:
            answer = self._chat_local(messages, temperature=temperature, max_tokens=max_tokens)
        if budget is not None:
            budget.check_resources()
        return answer

    # ------------------------------------------------------------------
    # API backend
    # ------------------------------------------------------------------
    def _ensure_api_client(self):
        if self._api_client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package is required for backend='api'. "
                "pip install openai"
            ) from e
        kwargs: dict[str, Any] = {"api_key": self.cfg.api_key}
        if self.cfg.base_url:
            kwargs["base_url"] = self.cfg.base_url
        kwargs["timeout"] = self.cfg.timeout
        self._api_client = OpenAI(**kwargs)

    def _chat_api(self, messages: list[dict], temperature: Optional[float], max_tokens: Optional[int] = None) -> str:
        self._ensure_api_client()
        temp = self.cfg.temperature if temperature is None else float(temperature)
        completion = self._api_client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=temp,
            max_tokens=self.cfg.max_tokens if max_tokens is None else max_tokens,
            timeout=self.cfg.timeout,
        )
        usage = getattr(completion, "usage", None)
        if usage is not None and getattr(usage, "total_tokens", None):
            self.total_tokens_used += int(usage.total_tokens)
        ans = completion.choices[0].message.content or ""
        self.last_response = ans
        return ans

    # ------------------------------------------------------------------
    # Local backend (lazy)
    # ------------------------------------------------------------------
    def _ensure_local_model(self):
        if self._local_model is not None:
            return
        try:
            self._load_local_model()
        except LocalModelInitializationError:
            raise
        except Exception as exc:
            raise LocalModelInitializationError(
                f"Cannot initialize local model {self.cfg.model!r}: {type(exc).__name__}: {exc}"
            ) from exc

    def _load_local_model(self):
        require_accelerate(self.cfg.device_map)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "transformers is required for backend='local'. "
                "pip install transformers torch"
            ) from e
        tok = AutoTokenizer.from_pretrained(self.cfg.model, trust_remote_code=True)
        if not getattr(tok, "chat_template", None):
            raise RuntimeError(
                "The DeepPrep checkpoint tokenizer has no chat_template; "
                "provide the official tokenizer files instead of a base model."
            )
        quantization = fp8_quantization_config(self.cfg.device_map)
        self._weight_quantization = "torchao_fp8_weight_only" if quantization is not None else None
        dtype = self.cfg.torch_dtype
        if dtype in ("auto", "bfloat16"):
            dtype = local_torch_dtype(self.cfg.device_map)
        mdl = AutoModelForCausalLM.from_pretrained(
            self.cfg.model,
            trust_remote_code=True,
            **model_dtype_kwargs(dtype),
            device_map=self.cfg.device_map,
            quantization_config=quantization,
        )
        mdl.eval()
        self._local_tokenizer = tok
        self._local_model = mdl

    def _chat_local(self, messages: list[dict], temperature: Optional[float], max_tokens: Optional[int] = None) -> str:
        self._ensure_local_model()
        tok = self._local_tokenizer
        mdl = self._local_model
        template_options = {}
        if self.cfg.enable_thinking is not None:
            template_options["enable_thinking"] = self.cfg.enable_thinking
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_options,
        )
        import torch  # noqa: WPS433  -- local-only import
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        temp = self.cfg.temperature if temperature is None else float(temperature)
        eos_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
        for marker in ("</expand>", "</answer>"):
            marker_ids = tok.encode(marker, add_special_tokens=False)
            if len(marker_ids) == 1 and marker_ids[0] not in eos_ids:
                eos_ids.append(marker_ids[0])
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=self.cfg.max_tokens if max_tokens is None else max_tokens,
                do_sample=temp > 0,
                temperature=max(temp, 0.01),
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
                pad_token_id=tok.eos_token_id,
                eos_token_id=eos_ids or None,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        # DeepPrep action markers may be registered as special tokens. They
        # must survive decoding so the protocol parser can see them.
        ans = tok.decode(gen_ids, skip_special_tokens=False)
        self.last_response = ans
        return ans

    # ------------------------------------------------------------------
    def attach_local_model(self, model, tokenizer) -> None:
        """Inject an externally loaded HF model (e.g. by ``RLTrainer``) so
        inference and training can share the same weights without reloading.
        """
        self._local_model = model
        self._local_tokenizer = tokenizer
        self.cfg.backend = "local"

    def runtime_metadata(self) -> dict:
        """Serializable checkpoint and generation metadata for run logs."""
        metadata = {
            "backend": self.cfg.backend,
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_new_tokens": self.cfg.max_tokens,
            "enable_thinking": self.cfg.enable_thinking,
            "repetition_penalty": self.cfg.repetition_penalty,
            "torch_dtype": self.cfg.torch_dtype,
            "weight_quantization": getattr(self, "_weight_quantization", None),
            "device_map": self.cfg.device_map,
            "total_tokens_used": self.total_tokens_used,
        }
        tok = self._local_tokenizer
        if tok is not None:
            metadata["chat_template"] = getattr(tok, "chat_template", None)
            metadata["special_tokens_map"] = getattr(tok, "special_tokens_map", {})
        return metadata


__all__ = ["LLMClient", "LLMConfig"]
