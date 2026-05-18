import logging
import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from helm_eval_lex_quirks.config import ModelConfig, model_config

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    completion: str
    logprobs: Optional[List]
    input_tokens: int = 0
    output_tokens: int = 0


def _parse_reset_seconds(s: str) -> Optional[float]:
    s = s.strip()
    m = re.fullmatch(r"([\d.]+)s", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except ValueError:
        return None


class AnthropicClient:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        request_interval: float = 2.0,
    ):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic support requires the 'anthropic' package. "
                "Run: pip install -r requirements.txt"
            ) from exc

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=0)
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
        for key in ("retry-after", "anthropic-ratelimit-requests-reset"):
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
            logger.debug("Anthropic Messages API does not support token logprobs; returning None.")

        max_attempts = 10
        for attempt in range(max_attempts):
            self._throttle()
            self._last_call_at = time.monotonic()

            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                usage = getattr(resp, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
                self._input_tokens += input_tokens
                self._output_tokens += output_tokens

                chunks = []
                for block in getattr(resp, "content", []) or []:
                    if getattr(block, "type", None) == "text":
                        chunks.append(getattr(block, "text", ""))
                return ModelResponse(
                    completion="\n".join(chunks).strip(),
                    logprobs=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except self._anthropic.RateLimitError as exc:
                headers = {}
                if hasattr(exc, "response") and exc.response is not None:
                    headers = dict(exc.response.headers)
                wait = self._wait_from_429_headers(headers, attempt)
                logger.warning(
                    "Anthropic rate limit (attempt %d/%d) — waiting %.0fs",
                    attempt + 1, max_attempts, wait,
                )
                time.sleep(wait)

            except self._anthropic.APIError as exc:
                logger.error("Anthropic API error on attempt %d: %s", attempt + 1, exc)
                if attempt == max_attempts - 1:
                    raise
                time.sleep(min(4 * (2 ** attempt), 30))

        raise RuntimeError(f"Anthropic call failed after {max_attempts} attempts")
