from dataclasses import dataclass, field
from typing import List, Optional

from compass.clients.anthropic import AnthropicClient as _CompassAnthropicClient

from compass_experiments.config import model_config


@dataclass
class ModelResponse:
    completion: str
    logprobs: Optional[List]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = ""
    request_settings: dict = field(default_factory=dict)
    billing_tokens: Optional[dict] = None
    usage_complete: bool = False
    usage_attempts: list = field(default_factory=list)

    @property
    def tokens_used(self) -> dict:
        return {"input": self.input_tokens, "output": self.output_tokens}


class AnthropicClient:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        request_interval: float = 2.0,
    ):
        self._inner = _CompassAnthropicClient(
            model=model, api_key=api_key, request_interval=request_interval
        )
        self.model = model
        self._config = model_config(model)

    @property
    def total_tokens(self):
        return self._inner.total_tokens

    @property
    def total_cost_usd(self) -> float:
        return self._inner.total_cost_usd

    def estimate_cost(self, prompts: List[str], max_tokens: int = 1) -> float:
        n_input = sum(len(p) // 4 for p in prompts)
        n_output = len(prompts) * max_tokens
        return (
            n_input * self._config.input_cost_per_million / 1_000_000
            + n_output * self._config.output_cost_per_million / 1_000_000
        )

    def request_settings(self, max_tokens, temperature):
        return self._inner.request_settings(max_tokens, temperature)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: bool = True,
        top_logprobs: int = 20,
        system: str = "You are a helpful assistant.",
    ) -> ModelResponse:
        resp = self._inner.complete(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature, system=system
        )
        tokens = resp.tokens_used or {}
        return ModelResponse(
            completion=resp.completion,
            logprobs=None,
            input_tokens=tokens.get("input", 0),
            output_tokens=tokens.get("output", 0),
            cost_usd=getattr(resp, "cost_usd", 0.0),
            finish_reason=str(getattr(resp, "finish_reason", "") or ""),
            request_settings=getattr(resp, "request_settings", {}),
            billing_tokens=getattr(resp, "billing_tokens", None),
            usage_complete=getattr(resp, "usage_complete", False),
            usage_attempts=getattr(resp, "usage_attempts", []),
        )
