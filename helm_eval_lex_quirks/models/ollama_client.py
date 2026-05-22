"""Ollama client for local LLM inference."""
import logging
from dataclasses import dataclass
from typing import List, Optional

try:
    from ollama import Client as OllamaAPIClient
except ImportError:
    raise ImportError(
        "ollama package required. Install with: pip install ollama"
    )

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    completion: str
    logprobs: Optional[List] = None
    input_tokens: int = 0
    output_tokens: int = 0


class OllamaClient:
    """Client for running models locally via Ollama."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        request_interval: float = 0.1,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., 'llama3.1:latest', 'mistral:latest')
            host: Ollama API endpoint
            request_interval: Minimum seconds between requests
        """
        self.api_client = OllamaAPIClient(host=host)
        self.model = model
        self.host = host
        self._input_tokens = 0
        self._output_tokens = 0
        self._request_interval = request_interval

    @property
    def total_tokens(self):
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def total_cost_usd(self) -> float:
        """Local models are free."""
        return 0.0

    def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> ModelResponse:
        """
        Generate completion via Ollama.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            system: Optional system prompt

        Returns:
            ModelResponse with completion text

        Raises:
            RuntimeError: If Ollama API call fails
        """
        # Build full prompt with system message if provided
        full_prompt = prompt
        if system:
            full_prompt = f"<system>\n{system}\n</system>\n\n{prompt}"

        try:
            # Call Ollama API
            response = self.api_client.generate(
                model=self.model,
                prompt=full_prompt,
                stream=False,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )

            completion = response.get("response", "")
            if not completion:
                raise RuntimeError(f"Empty response from {self.model}")

            # Ollama doesn't expose token counts directly,
            # but we can estimate from response length
            estimated_output_tokens = len(completion.split()) * 1.3
            self._output_tokens += int(estimated_output_tokens)

            return ModelResponse(
                completion=completion,
                logprobs=None,
                input_tokens=len(prompt.split()),
                output_tokens=int(estimated_output_tokens),
            )

        except Exception as e:
            logger.error(f"Ollama error for model {self.model}: {e}")
            raise RuntimeError(f"Ollama inference failed for {self.model}: {e}") from e
