import logging
import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import openai

from helm_eval_lex_quirks.config import ModelConfig, model_config

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    completion: str
    logprobs: Optional[List]
    input_tokens: int = 0
    output_tokens: int = 0


def _parse_reset_seconds(s: str) -> Optional[float]:
    """Parse OpenAI's rate limit reset values: '20s', '1m0s', '1m30.5s'."""
    s = s.strip()
    m = re.fullmatch(r"([\d.]+)s", s)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(\d+)m([\d.]+)s", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.fullmatch(r"(\d+)m", s)
    if m:
        return int(m.group(1)) * 60
    try:
        return float(s)
    except ValueError:
        return None


class OpenAIResponsesClient:
    """OpenAI Responses API client for GPT-5 models."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        request_interval: float = 2.0,
    ):
        self.client = openai.OpenAI(api_key=api_key, max_retries=0)
        self.model = model
        self.config: ModelConfig = model_config(model)
        self._input_tokens = 0
        self._output_tokens = 0
        self._request_interval = request_interval
        self._last_call_at: float = 0.0

    @property
    def total_tokens(self):
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        return (
            self._input_tokens * self.config.input_cost_per_million / 1_000_000
            + self._output_tokens * self.config.output_cost_per_million / 1_000_000
        )

    def estimate_cost(self, prompts: List[str], max_tokens: int = 1) -> float:
        # Note: runner.estimate_model_cost() already applies 10x multiplier for gpt-5
        n_input = sum(len(p) // 4 for p in prompts)
        n_output = len(prompts) * max_tokens
        return (
            n_input * self.config.input_cost_per_million / 1_000_000
            + n_output * self.config.output_cost_per_million / 1_000_000
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        gap = self._request_interval - elapsed
        if gap > 0:
            time.sleep(gap)

    def _wait_from_429_headers(self, headers: dict, attempt: int) -> float:
        for key in ("retry-after", "x-ratelimit-reset-requests"):
            val = headers.get(key)
            if val:
                secs = _parse_reset_seconds(val)
                if secs is not None:
                    return secs + random.uniform(0.5, 2.0)
        return min(15 * (2 ** attempt), 60) + random.uniform(0, 5)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: bool = True,
        top_logprobs: int = 20,
        system: str = "You are a helpful assistant.",
    ) -> ModelResponse:
        if logprobs:
            logger.debug("Responses API does not support logprobs; returning None.")

        max_attempts = 10
        for attempt in range(max_attempts):
            self._throttle()
            self._last_call_at = time.monotonic()

            try:
                # GPT-5 models need larger token budgets for reasoning + output
                # Use 10x for reasoning models to ensure both reasoning and text generation
                actual_max_tokens = max_tokens * 10 if self.model.startswith("gpt-5") else max_tokens

                resp = self.client.responses.create(
                    model=self.model,
                    instructions=system,
                    input=prompt,
                    max_output_tokens=actual_max_tokens,
                )

                # Read token usage immediately (even if we might retry)
                usage = getattr(resp, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
                    self._input_tokens += input_tokens
                    self._output_tokens += output_tokens

                completion = resp.output_text or ""

                # Reject empty outputs to prevent data contamination (but tokens already counted)
                if not completion.strip():
                    logger.warning(
                        "Responses API returned empty output (tokens counted). status=%s, "
                        "incomplete_details=%s, usage=%s, attempt=%d/%d",
                        getattr(resp, "status", "unknown"),
                        getattr(resp, "incomplete_details", None),
                        usage,
                        attempt + 1, max_attempts,
                    )
                    if attempt == max_attempts - 1:
                        raise RuntimeError(f"Responses API returned empty output after {max_attempts} attempts")
                    time.sleep(min(4 * (2 ** attempt), 30))
                    continue

                # Fallback token estimation if usage unavailable
                if not usage:
                    input_tokens = 0
                    output_tokens = len(completion) // 4
                    self._output_tokens += output_tokens

                return ModelResponse(
                    completion=completion,
                    logprobs=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except openai.RateLimitError as exc:
                headers: dict = {}
                if hasattr(exc, "response") and exc.response is not None:
                    headers = dict(exc.response.headers)
                wait = self._wait_from_429_headers(headers, attempt)
                logger.warning(
                    "Rate limit (attempt %d/%d) — waiting %.0fs",
                    attempt + 1, max_attempts, wait,
                )
                time.sleep(wait)

            except openai.APIError as exc:
                logger.error("API error on attempt %d: %s", attempt + 1, exc)
                if attempt == max_attempts - 1:
                    raise
                time.sleep(min(4 * (2 ** attempt), 30))

        raise RuntimeError(f"Responses API call failed after {max_attempts} attempts")
