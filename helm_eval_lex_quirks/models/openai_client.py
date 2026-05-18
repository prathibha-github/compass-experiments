import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import openai

from helm_eval_lex_quirks.config import MODEL_CONFIGS, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    completion: str
    logprobs: Optional[List]
    input_tokens: int = 0
    output_tokens: int = 0


def _parse_reset_seconds(s: str) -> Optional[float]:
    """Parse OpenAI's x-ratelimit-reset-* values: '20s', '1m0s', '1m30.5s'."""
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


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        request_interval: float = 2.0,
    ):
        # max_retries=0: the SDK must not absorb 429s before we handle them.
        self.client   = openai.OpenAI(api_key=api_key, max_retries=0)
        self.model    = model
        self.config: ModelConfig = MODEL_CONFIGS.get(model, MODEL_CONFIGS["gpt-4o-mini"])
        self._input_tokens    = 0
        self._output_tokens   = 0
        self._request_interval = request_interval
        self._last_call_at: float = 0.0

    # ── cost tracking ────────────────────────────────────────────────────────

    @property
    def total_tokens(self):
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        return (
            self._input_tokens  * self.config.input_cost_per_million  / 1_000_000
            + self._output_tokens * self.config.output_cost_per_million / 1_000_000
        )

    def estimate_cost(self, prompts: List[str], max_tokens: int = 1) -> float:
        n_input  = sum(len(p) // 4 for p in prompts)
        n_output = len(prompts) * max_tokens
        return (
            n_input  * self.config.input_cost_per_million  / 1_000_000
            + n_output * self.config.output_cost_per_million / 1_000_000
        )

    # ── throttling / header helpers ──────────────────────────────────────────

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        gap     = self._request_interval - elapsed
        if gap > 0:
            time.sleep(gap)

    def _adapt_from_success_headers(self, headers: dict) -> None:
        """Proactively pause when quota is nearly exhausted."""
        for quota_type in ("requests", "tokens"):
            remaining = headers.get(f"x-ratelimit-remaining-{quota_type}")
            reset_str  = headers.get(f"x-ratelimit-reset-{quota_type}")
            if remaining is None or reset_str is None:
                continue
            try:
                if int(remaining) <= 1:
                    wait = (_parse_reset_seconds(reset_str) or 60) + 1.0
                    logger.info(
                        "Quota for %s nearly exhausted — pausing %.0fs for window reset",
                        quota_type, wait,
                    )
                    time.sleep(wait)
                    break
            except (ValueError, TypeError):
                continue

    def _wait_from_429_headers(self, headers: dict, attempt: int) -> float:
        for key in ("retry-after", "x-ratelimit-reset-requests"):
            val = headers.get(key)
            if val:
                secs = _parse_reset_seconds(val)
                if secs is not None:
                    return secs + random.uniform(0.5, 2.0)
        return min(15 * (2 ** attempt), 60) + random.uniform(0, 5)

    # ── inference ────────────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: bool = True,
        top_logprobs: int = 20,   # 20 = API max; ensures A/B/C/D are covered
        system: str = "You are a helpful assistant.",
    ) -> ModelResponse:
        kwargs: dict = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if logprobs:
            kwargs["logprobs"]     = True
            kwargs["top_logprobs"] = top_logprobs

        max_attempts = 10
        for attempt in range(max_attempts):
            self._throttle()
            self._last_call_at = time.monotonic()

            try:
                # with_raw_response is the only way to read response headers
                # from the openai SDK; the parsed response object does not expose them.
                raw  = self.client.chat.completions.with_raw_response.create(**kwargs)
                resp = raw.parse()
                headers = dict(raw.headers)

                usage = resp.usage
                self._input_tokens  += usage.prompt_tokens
                self._output_tokens += usage.completion_tokens

                self._adapt_from_success_headers(headers)

                raw_lp = None
                if logprobs and resp.choices[0].logprobs:
                    content = resp.choices[0].logprobs.content
                    if content:
                        raw_lp = content[0].top_logprobs

                return ModelResponse(
                    completion=resp.choices[0].message.content.strip(),
                    logprobs=raw_lp,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                )

            except openai.RateLimitError as exc:
                headers: dict = {}
                if hasattr(exc, "response") and exc.response is not None:
                    headers = dict(exc.response.headers)
                wait = self._wait_from_429_headers(headers, attempt)
                logger.warning(
                    "Rate limit (attempt %d/%d) — waiting %.0fs  "
                    "[retry-after=%s  reset-requests=%s]",
                    attempt + 1, max_attempts, wait,
                    headers.get("retry-after", "—"),
                    headers.get("x-ratelimit-reset-requests", "—"),
                )
                time.sleep(wait)

            except openai.APIError as exc:
                logger.error("API error on attempt %d: %s", attempt + 1, exc)
                if attempt == max_attempts - 1:
                    raise
                time.sleep(min(4 * (2 ** attempt), 30))

        raise RuntimeError(f"OpenAI call failed after {max_attempts} attempts")
