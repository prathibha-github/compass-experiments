from typing import Protocol

from helm_eval_lex_quirks.config import EvalConfig, model_provider
from helm_eval_lex_quirks.models.anthropic_client import AnthropicClient
from helm_eval_lex_quirks.models.gemini_client import GeminiClient
from helm_eval_lex_quirks.models.openai_client import OpenAIClient
from helm_eval_lex_quirks.models.openai_responses_client import OpenAIResponsesClient


class CompletionClient(Protocol):
    model: str

    @property
    def total_tokens(self): ...

    @property
    def total_cost_usd(self) -> float: ...

    def estimate_cost(self, prompts, max_tokens: int = 1) -> float: ...

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: bool = True,
        top_logprobs: int = 20,
        system: str = "You are a helpful assistant.",
    ): ...


def create_client(
    config: EvalConfig,
    model_name: str,
    disable_safety_filters: bool = False,
) -> CompletionClient:
    provider = model_provider(model_name)
    if provider == "anthropic":
        if not config.anthropic_api_key:
            raise ValueError(
                f"Model {model_name!r} requires --anthropic-api-key or ANTHROPIC_API_KEY."
            )
        return AnthropicClient(config.anthropic_api_key, model_name, config.request_interval)

    if provider == "google":
        if not config.google_api_key:
            raise ValueError(
                f"Model {model_name!r} requires --google-api-key or GOOGLE_API_KEY."
            )
        return GeminiClient(
            config.google_api_key, model_name, config.request_interval,
            disable_safety_filters=disable_safety_filters,
        )

    # GPT-5 models use the Responses API
    if model_name.startswith("gpt-5"):
        if not config.api_key:
            raise ValueError(f"Model {model_name!r} requires --api-key or OPENAI_API_KEY.")
        return OpenAIResponsesClient(config.api_key, model_name, config.request_interval)

    if not config.api_key:
        raise ValueError(f"Model {model_name!r} requires --api-key or OPENAI_API_KEY.")
    return OpenAIClient(config.api_key, model_name, config.request_interval)
