import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from compass.clients.openai import OpenAIClient as _CompassOpenAIClient

from helm_eval_lex_quirks.config import model_config

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    completion: str
    logprobs: Optional[List]
    input_tokens: int = 0
    output_tokens: int = 0


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        request_interval: float = 2.0,
    ):
        self._inner = _CompassOpenAIClient(
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

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: bool = True,
        top_logprobs: int = 20,
        system: str = "You are a helpful assistant.",
    ) -> ModelResponse:
        if not logprobs:
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

        # Logprob path: call compass's underlying openai.OpenAI client directly so
        # ECE calibration (runner.py) gets real token probabilities. Reuse compass's
        # throttle, token counters, quota-header adaptation, and retry helpers so this
        # path is as reliable as the normal compass inference path.
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        actual_temperature = (
            1.0 if (self.model.startswith("gpt-5") or self.model.startswith("o4"))
            else temperature
        )

        kwargs = dict(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=actual_temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
        )

        _openai = self._inner._openai
        max_attempts = 10
        for attempt in range(max_attempts):
            self._inner._throttle()
            self._inner._last_call_at = time.monotonic()
            try:
                raw = self._inner.client.chat.completions.with_raw_response.create(**kwargs)
                resp = raw.parse()
                headers = dict(raw.headers)

                completion = resp.choices[0].message.content or ""
                raw_logprobs = None
                choice_lp = resp.choices[0].logprobs
                if choice_lp and choice_lp.content:
                    raw_logprobs = choice_lp.content[0].top_logprobs

                usage = resp.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                self._inner._input_tokens += input_tokens
                self._inner._output_tokens += output_tokens

                self._inner._adapt_from_success_headers(headers)

                return ModelResponse(
                    completion=completion,
                    logprobs=raw_logprobs,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except _openai.RateLimitError as exc:
                headers_429: dict = {}
                if hasattr(exc, "response") and exc.response is not None:
                    headers_429 = dict(exc.response.headers)
                wait = self._inner._wait_from_429_headers(headers_429, attempt)
                logger.warning(
                    "Rate limit on logprob call (attempt %d/%d) — waiting %.0fs",
                    attempt + 1, max_attempts, wait,
                )
                time.sleep(wait)

            except _openai.APIError as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise RuntimeError(
                        f"OpenAI logprob call failed for {self.model}: {exc}"
                    ) from exc
                logger.error("OpenAI API error on logprob attempt %d: %s", attempt + 1, exc)
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"OpenAI logprob call failed for {self.model}: {exc}"
                    ) from exc
                time.sleep(min(4 * (2 ** attempt), 30))

        raise RuntimeError(f"OpenAI logprob call failed after {max_attempts} attempts")
