"""Unit tests for create_client routing and key validation behavior."""

import unittest
from unittest.mock import patch

from helm_eval_lex_quirks.models import create_client
from tests._test_base import make_eval_config


class ModelFactoryTests(unittest.TestCase):
    def _config(self, **overrides):
        base = {"api_key": "openai-key", "output_dir": "results"}
        base.update(overrides)
        return make_eval_config(**base)

    def test_create_client_requires_anthropic_key(self):
        cfg = self._config(anthropic_api_key=None)
        with self.assertRaisesRegex(ValueError, "requires --anthropic-api-key"):
            create_client(cfg, "claude-haiku-4-5-20251001")

    def test_create_client_requires_openai_key(self):
        cfg = self._config(api_key=None)
        with self.assertRaisesRegex(ValueError, "requires --api-key"):
            create_client(cfg, "gpt-4o-mini")

    def test_create_client_routes_gpt5_to_responses_client(self):
        cfg = self._config()
        with patch("helm_eval_lex_quirks.models.OpenAIResponsesClient", return_value="responses") as patched:
            out = create_client(cfg, "gpt-5-mini")
        self.assertEqual(out, "responses")
        patched.assert_called_once_with("openai-key", "gpt-5-mini", cfg.request_interval)

    def test_create_client_routes_standard_openai_model(self):
        cfg = self._config()
        with patch("helm_eval_lex_quirks.models.OpenAIClient", return_value="chat") as patched:
            out = create_client(cfg, "gpt-4o-mini")
        self.assertEqual(out, "chat")
        patched.assert_called_once_with("openai-key", "gpt-4o-mini", cfg.request_interval)

    def test_create_client_routes_anthropic_model(self):
        cfg = self._config()
        with patch("helm_eval_lex_quirks.models.AnthropicClient", return_value="anthropic") as patched:
            out = create_client(cfg, "claude-haiku-4-5-20251001")
        self.assertEqual(out, "anthropic")
        patched.assert_called_once_with("anthropic-key", "claude-haiku-4-5-20251001", cfg.request_interval)

    def test_create_client_requires_google_key(self):
        cfg = self._config(google_api_key=None)
        with self.assertRaisesRegex(ValueError, "requires --google-api-key"):
            create_client(cfg, "gemini-2.0-flash")

    def test_create_client_routes_gemini_model(self):
        cfg = self._config(google_api_key="fake-google-key")
        with patch("helm_eval_lex_quirks.models.GeminiClient", return_value="gemini") as patched:
            out = create_client(cfg, "gemini-2.0-flash")
        self.assertEqual(out, "gemini")
        patched.assert_called_once_with(
            "fake-google-key", "gemini-2.0-flash", cfg.request_interval,
            disable_safety_filters=False,
        )

    def test_create_client_routes_gemini_model_with_safety_override(self):
        cfg = self._config(google_api_key="fake-google-key")
        with patch("helm_eval_lex_quirks.models.GeminiClient", return_value="gemini") as patched:
            out = create_client(cfg, "gemini-2.0-flash", disable_safety_filters=True)
        self.assertEqual(out, "gemini")
        patched.assert_called_once_with(
            "fake-google-key", "gemini-2.0-flash", cfg.request_interval,
            disable_safety_filters=True,
        )


if __name__ == "__main__":
    unittest.main()
