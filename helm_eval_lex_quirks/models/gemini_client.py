import time
from dataclasses import dataclass
from typing import List, Optional

from google.genai import types as _genai_types

from compass.clients.google_ai import GoogleAIClient as _CompassGoogleAIClient

from helm_eval_lex_quirks.config import model_config

# Used only when disable_safety_filters=True (e.g. SELF_HARM_SAFETY_SUITE), so that
# Gemini's content filters do not block the test prompts before the model can respond.
# Do NOT apply this globally — it changes the safety configuration vs. default provider
# behavior and distorts cross-model comparisons on non-safety suites.
_SAFETY_OFF = [
    _genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    _genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",         threshold="BLOCK_NONE"),
    _genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",        threshold="BLOCK_NONE"),
    _genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="BLOCK_NONE"),
]


@dataclass
class ModelResponse:
    completion: str
    logprobs: Optional[List] = None
    input_tokens: int = 0
    output_tokens: int = 0


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        request_interval: float = 2.0,
        disable_safety_filters: bool = False,
    ):
        self._inner = _CompassGoogleAIClient(
            model=model, api_key=api_key, request_interval=request_interval
        )
        self.model = model
        self._config = model_config(model)
        self._request_count = 0
        self._disable_safety_filters = disable_safety_filters

    @property
    def total_tokens(self):
        return self._inner.total_tokens

    @property
    def total_cost_usd(self) -> float:
        tokens = self._inner.total_tokens
        return (
            tokens["input"] * self._config.input_cost_per_million / 1_000_000
            + tokens["output"] * self._config.output_cost_per_million / 1_000_000
        )

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
        logprobs: bool = True,
        top_logprobs: int = 20,
        system: str = "You are a helpful assistant.",
    ) -> ModelResponse:
        if self._config.max_requests and self._request_count >= self._config.max_requests:
            raise RuntimeError(
                f"Reached max_requests={self._config.max_requests} for {self.model}. "
                "Increase max_requests in config.py or use a paid-tier key."
            )
        self._request_count += 1

        # Bypass compass complete() to control safety_settings. Use compass throttle and
        # token counters so total_tokens / total_cost_usd stay accurate.
        self._inner._throttle()
        self._inner._last_call_at = time.monotonic()

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        config_kwargs: dict = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._disable_safety_filters:
            config_kwargs["safety_settings"] = _SAFETY_OFF

        try:
            response = self._inner.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=_genai_types.GenerateContentConfig(**config_kwargs),
            )
            completion = response.text if response.text else ""
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                input_tokens = getattr(usage, "prompt_token_count", None) or 0
                output_tokens = getattr(usage, "candidates_token_count", None) or 0
            else:
                input_tokens = len(full_prompt.split())
                output_tokens = len(completion.split())

            self._inner._input_tokens += input_tokens
            self._inner._output_tokens += output_tokens

            return ModelResponse(
                completion=completion.strip(),
                logprobs=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini inference failed: {exc}") from exc
