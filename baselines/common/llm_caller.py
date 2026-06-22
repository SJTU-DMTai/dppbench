import os
from openai import OpenAI

from baselines.common.api_keys import load_openai_compatible_api_config


class LLMCaller:
    def __init__(self, model=None, api_key=None, base_url=None, temperature=0.7):
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o")
        if api_key is not None or base_url is not None:
            raise ValueError(
                "LLM API credentials must be loaded from apikeys.json; "
                "do not pass api_key/base_url through code or CLI."
            )
        api_config = load_openai_compatible_api_config()
        api_key = api_config["api_key"]
        base_url = api_config.get("base_url")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.temperature = temperature
        self.total_tokens_used = 0

    def query(self, messages, temperature=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
        )
        usage = response.usage
        if usage:
            self.total_tokens_used += usage.total_tokens
        return response.choices[0].message.content
