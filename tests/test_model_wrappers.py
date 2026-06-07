"""Unit tests for thin provider wrappers around Compass clients."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from helm_eval_lex_quirks.models.gemini_client import GeminiClient
from helm_eval_lex_quirks.models.openai_client import OpenAIClient
from helm_eval_lex_quirks.models.openai_responses_client import OpenAIResponsesClient


class OpenAIWrapperTests(unittest.TestCase):
    def test_openai_wrapper_delegates_logprobs_to_compass(self):
        inner = MagicMock()
        inner.complete.return_value = SimpleNamespace(
            completion="A",
            logprobs=[{"token": "A", "logprob": -0.1}],
            tokens_used={"input": 12, "output": 3},
        )
        inner.total_tokens = {"input": 12, "output": 3}
        inner.total_cost_usd = 0.02

        with patch(
            "helm_eval_lex_quirks.models.openai_client._CompassOpenAIClient",
            return_value=inner,
        ):
            client = OpenAIClient("openai-key", "gpt-4o-mini", request_interval=0.5)
            response = client.complete(
                "prompt",
                max_tokens=5,
                temperature=0.2,
                logprobs=True,
                top_logprobs=7,
                system="system",
            )

        inner.complete.assert_called_once_with(
            prompt="prompt",
            max_tokens=5,
            temperature=0.2,
            system="system",
            logprobs=True,
            top_logprobs=7,
        )
        self.assertEqual(response.completion, "A")
        self.assertEqual(response.logprobs, [{"token": "A", "logprob": -0.1}])
        self.assertEqual(response.input_tokens, 12)
        self.assertEqual(response.output_tokens, 3)


class OpenAIResponsesWrapperTests(unittest.TestCase):
    def test_gpt5_uses_required_temperature_and_output_multiplier(self):
        inner = MagicMock()
        inner.complete.return_value = SimpleNamespace(
            completion="answer",
            tokens_used={"input": 3, "output": 2},
        )

        with patch(
            "helm_eval_lex_quirks.models.openai_responses_client._CompassOpenAIResponsesClient",
            return_value=inner,
        ) as patched:
            client = OpenAIResponsesClient("openai-key", "gpt-5-mini", 0.5)
            response = client.complete(
                "prompt", max_tokens=200, temperature=0.0, system="system"
            )

        patched.assert_called_once_with(
            model="gpt-5-mini",
            api_key="openai-key",
            request_interval=0.5,
            max_output_token_multiplier=10,
            required_temperature=1.0,
        )
        inner.complete.assert_called_once_with(
            prompt="prompt",
            max_tokens=200,
            temperature=0.0,
            system="system",
        )
        self.assertEqual(response.completion, "answer")
        self.assertIsNone(response.logprobs)
        self.assertEqual(response.input_tokens, 3)
        self.assertEqual(response.output_tokens, 2)

    def test_non_gpt5_responses_model_uses_default_output_multiplier(self):
        with patch(
            "helm_eval_lex_quirks.models.openai_responses_client._CompassOpenAIResponsesClient",
        ) as patched:
            OpenAIResponsesClient("openai-key", "gpt-4o-mini", 0.5)

        patched.assert_called_once_with(
            model="gpt-4o-mini",
            api_key="openai-key",
            request_interval=0.5,
            max_output_token_multiplier=1,
            required_temperature=None,
        )


class GeminiWrapperTests(unittest.TestCase):
    def test_gemini_wrapper_forwards_safety_override_and_cost(self):
        inner = MagicMock()
        inner.complete.return_value = SimpleNamespace(
            completion="answer",
            logprobs=None,
            tokens_used={"input": 9, "output": 4},
        )
        inner.total_tokens = {"input": 9, "output": 4}
        inner.total_cost_usd = 0.015

        with patch(
            "helm_eval_lex_quirks.models.gemini_client._CompassGoogleAIClient",
            return_value=inner,
        ) as patched:
            client = GeminiClient(
                "google-key",
                "gemini-2.0-flash",
                request_interval=0.25,
                disable_safety_filters=True,
            )
            response = client.complete("prompt", max_tokens=4, system="system")

        patched.assert_called_once_with(
            model="gemini-2.0-flash",
            api_key="google-key",
            request_interval=0.25,
            disable_safety_filters=True,
        )
        inner.complete.assert_called_once_with(
            prompt="prompt",
            max_tokens=4,
            temperature=0.0,
            system="system",
            logprobs=True,
            top_logprobs=20,
        )
        self.assertEqual(client.total_cost_usd, 0.015)
        self.assertEqual(response.completion, "answer")
        self.assertEqual(response.input_tokens, 9)
        self.assertEqual(response.output_tokens, 4)


if __name__ == "__main__":
    unittest.main()
