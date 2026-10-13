"""BAT call traces survive parser failures and exhausted search budgets."""
import json
import time

from baselines.DeepPrep.llm_client import LLMClient
from baselines.common.budget import BudgetExceededError, get_active_budget


class BATSearchTimeLimit(BudgetExceededError):
    """Stop new generation before the remaining search time is exhausted."""


class BATLLMClient(LLMClient):
    def __init__(self, *, trace_path, **kwargs):
        super().__init__(**kwargs)
        self.trace_path = trace_path
        self._estimated_call_seconds = 150.0

    def chat(self, messages, **kwargs):
        budget = get_active_budget()
        if budget and budget.max_seconds is not None:
            remaining = budget.max_seconds - budget.elapsed_seconds
            if remaining < self._estimated_call_seconds * 1.5 + 30.0:
                raise BATSearchTimeLimit(f"insufficient time for another BAT call: {remaining:.1f}s remain")
        started = time.monotonic()
        record = {"time": time.time(), "messages": messages, "options": kwargs}
        self.last_response = None
        try:
            response = super().chat(messages, **kwargs)
            record["response"] = response
            return response
        except Exception as exc:
            record["response"] = self.last_response
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._estimated_call_seconds = max(self._estimated_call_seconds,
                                               time.monotonic() - started)
            with open(self.trace_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
