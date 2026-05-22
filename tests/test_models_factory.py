import unittest
from unittest.mock import patch

from helm_eval_lex_quirks.config import EvalConfig
from helm_eval_lex_quirks.models import create_client


class ModelFactoryTests(unittest.TestCase):
    def _config(self, **overrides):
        base = dict(
            api_key="openai-key",
            anthropic_api_key="anthropic-key",
            output_dir="results",
            scenarios=[],
            style_only=True,
        )
        base.update(overrides)
        return EvalConfig(**base)

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


if __name__ == "__main__":
    unittest.main()
