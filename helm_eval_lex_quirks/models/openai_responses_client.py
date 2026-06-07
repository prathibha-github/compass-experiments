from dataclasses import dataclass
from typing import List, Optional

from compass.clients.openai_responses import OpenAIResponsesClient as _CompassOpenAIResponsesClient

from helm_eval_lex_quirks.config import model_config


@dataclass
class ModelResponse:
    completion: str
    logprobs: Optional[List]
    input_tokens: int = 0
    output_tokens: int = 0


class OpenAIResponsesClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        request_interval: float = 2.0,
    ):
        # GPT-5 models reject temperatures other than 1.0 at the API level.
        required_temp = 1.0 if model.startswith("gpt-5") else None
        # GPT-5 reasoning can consume hidden reasoning tokens before visible output.
        max_output_token_multiplier = 10 if model.startswith("gpt-5") else 1
        self._inner = _CompassOpenAIResponsesClient(
            model=model,
            api_key=api_key,
            request_interval=request_interval,
            max_output_token_multiplier=max_output_token_multiplier,
            required_temperature=required_temp,
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

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 0,
        system: str = "You are a helpful assistant.",
    ) -> ModelResponse:
        # Responses API does not support logprobs; always returns None.
        # Callers that rely on logprobs (e.g. ECE) will get coverage=0.
        resp = self._inner.complete(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature, system=system
        )
        tokens = resp.tokens_used or {}
        return ModelResponse(
            completion=resp.completion,
            logprobs=None,
            input_tokens=tokens.get("input", 0),
            output_tokens=tokens.get("output", 0),
        )
