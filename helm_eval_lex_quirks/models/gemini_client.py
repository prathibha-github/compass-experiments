import logging
import random
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


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        request_interval: float = 2.0,
    ):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Gemini support requires the 'google-genai' package. "
                "Run: pip install google-genai"
            ) from exc

        self._genai = genai
        # Use per-instance client configuration (not global genai.configure)
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.config: ModelConfig = model_config(model)
        self._input_tokens = 0
        self._output_tokens = 0
        self._request_count = 0
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

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: bool = True,
        top_logprobs: int = 20,
        system: str = "You are a helpful assistant.",
    ) -> ModelResponse:
        if self.config.max_requests is not None and self._request_count >= self.config.max_requests:
            raise RuntimeError(
                f"Model {self.model!r} has reached its maximum request limit of {self.config.max_requests}"
            )

        if system != "You are a helpful assistant.":
            logger.warning(
                "Gemini client uses a fixed default system instruction. "
                "Per-call system prompts are not supported and will be ignored."
            )

        if logprobs:
            logger.debug("Gemini API does not support token logprobs; returning None.")

        max_attempts = 10
        for attempt in range(max_attempts):
            self._throttle()
            self._last_call_at = time.monotonic()

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self._genai.types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        system_instruction=system,
                        safety_settings=[
                            self._genai.types.SafetySetting(
                                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                threshold="BLOCK_NONE",
                            ),
                            self._genai.types.SafetySetting(
                                category="HARM_CATEGORY_HATE_SPEECH",
                                threshold="BLOCK_NONE",
                            ),
                            self._genai.types.SafetySetting(
                                category="HARM_CATEGORY_HARASSMENT",
                                threshold="BLOCK_NONE",
                            ),
                            self._genai.types.SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="BLOCK_NONE",
                            ),
                        ],
                    ),
                )

                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
                self._input_tokens += input_tokens
                self._output_tokens += output_tokens
                self._request_count += 1

                # Check if response has valid candidates before accessing text property,
                # which raises ValueError when all candidates are filtered out.
                if response.candidates and len(response.candidates) > 0:
                    completion = response.text
                else:
                    completion = ""

                return ModelResponse(
                    completion=completion.strip(),
                    logprobs=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except self._genai.types.BlockedPromptException as exc:
                logger.error("Gemini API blocked prompt: %s", exc)
                raise

            except Exception as exc:
                error_name = type(exc).__name__
                # Detect rate limit errors: google.api_core.exceptions.ResourceExhausted (429)
                is_rate_limit = "ResourceExhausted" in error_name or "429" in str(exc)

                if is_rate_limit:
                    wait = min(15 * (2 ** attempt), 60) + random.uniform(0, 5)
                    logger.warning(
                        "Gemini rate limit (attempt %d/%d) — waiting %.0fs",
                        attempt + 1, max_attempts, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Gemini API error on attempt %d: %s", attempt + 1, exc)
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(min(4 * (2 ** attempt), 30))

        raise RuntimeError(f"Gemini call failed after {max_attempts} attempts")
