"""Helpers for loading LLM API credentials from project-level apikeys.json."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


API_KEYS_FILE = "apikeys.json"
DEFAULT_PROVIDER = "DeepSeek"


def repo_root() -> Path:
    """Return the repository root from this common baseline module."""
    return Path(__file__).resolve().parents[2]


def api_keys_path(api_keys_path_override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the apikeys.json path.

    ``DPPBENCH_API_KEYS_PATH`` may point to an alternate file with the same
    schema. It changes only the credential file location, not the credential
    values themselves.
    """
    if api_keys_path_override:
        return Path(api_keys_path_override).expanduser().resolve()
    env_path = os.environ.get("DPPBENCH_API_KEYS_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return repo_root() / API_KEYS_FILE


def load_api_keys(api_keys_path_override: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the provider mapping from apikeys.json."""
    path = api_keys_path(api_keys_path_override)
    if not path.is_file():
        raise FileNotFoundError(
            f"LLM API credential file not found: {path}. "
            "Create apikeys.json at the project root."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"LLM API credential file must be a JSON object: {path}")
    return data


def load_openai_compatible_api_config(
    provider: str | None = None,
    api_keys_path_override: str | os.PathLike[str] | None = None,
) -> dict[str, str | None]:
    """Load OpenAI-compatible API config for one provider.

    Expected provider schema:

    ```json
    {
      "ProviderName": {
        "OPENAI_API_KEY": "...",
        "OPENAI_BASE_URL": "https://..."
      }
    }
    ```
    """
    data = load_api_keys(api_keys_path_override)
    provider_name = provider or os.environ.get("DPPBENCH_LLM_PROVIDER")
    if provider_name:
        raw_config = data.get(provider_name)
        if raw_config is None:
            raise KeyError(
                f"Provider {provider_name!r} not found in {api_keys_path(api_keys_path_override)}"
            )
    elif DEFAULT_PROVIDER in data:
        provider_name = DEFAULT_PROVIDER
        raw_config = data[provider_name]
    elif len(data) == 1:
        provider_name, raw_config = next(iter(data.items()))
    else:
        raise KeyError(
            "Multiple LLM providers are configured; set DPPBENCH_LLM_PROVIDER "
            "or use the default provider name."
        )

    if not isinstance(raw_config, dict):
        raise ValueError(f"Provider {provider_name!r} config must be a JSON object")

    api_key = raw_config.get("OPENAI_API_KEY")
    base_url = raw_config.get("OPENAI_BASE_URL")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(
            f"Provider {provider_name!r} must define a non-empty OPENAI_API_KEY"
        )
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError(f"Provider {provider_name!r} OPENAI_BASE_URL must be a string")

    return {
        "provider": provider_name,
        "api_key": api_key,
        "base_url": base_url,
    }
