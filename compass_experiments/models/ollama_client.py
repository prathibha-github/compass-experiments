from dataclasses import dataclass, field
from typing import List, Optional

from compass.clients.ollama import OllamaClient as _CompassOllamaClient


@dataclass
class ModelResponse:
    completion: str
    logprobs: Optional[List] = None
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


class OllamaClient:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        request_interval: float = 0.1,
    ):
        self._inner = _CompassOllamaClient(
            model=model, host=host, request_interval=request_interval
        )
        self.model = model

    @property
    def total_tokens(self):
        return self._inner.total_tokens

    @property
    def total_cost_usd(self) -> float:
        return self._inner.total_cost_usd

    def request_settings(self, max_tokens, temperature):
        return self._inner.request_settings(max_tokens, temperature)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system: Optional[str] = None,
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
