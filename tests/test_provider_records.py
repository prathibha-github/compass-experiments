import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class ProviderAdapterTests(unittest.TestCase):
    def test_all_adapters_preserve_provider_records(self):
        cases = [
            ('anthropic_client', 'AnthropicClient', '_CompassAnthropicClient', 'claude-haiku-4-5'),
            ('gemini_client', 'GeminiClient', '_CompassGoogleAIClient', 'gemini-2.0-flash'),
            ('openai_client', 'OpenAIClient', '_CompassOpenAIClient', 'o4-mini'),
            ('openai_responses_client', 'OpenAIResponsesClient', '_CompassOpenAIResponsesClient', 'gpt-5-mini'),
            ('ollama_client', 'OllamaClient', '_CompassOllamaClient', 'llama3.2'),
        ]
        settings = {'requested': {'max_tokens': 20, 'temperature': 0},
                    'effective': {'max_tokens': 200, 'temperature': 1}}
        response = SimpleNamespace(completion='answer', logprobs=None,
            tokens_used={'input': 10, 'output': 20}, cost_usd=.001,
            finish_reason='max_output_tokens', request_settings=settings,
            billing_tokens={'input': 30, 'output': 40}, usage_complete=True,
            usage_attempts=[{'cost_usd': .001}])
        for module_name, class_name, inner_name, model in cases:
            with self.subTest(provider=module_name):
                module = importlib.import_module('compass_experiments.models.' + module_name)
                inner = Mock(complete=Mock(return_value=response), request_settings=Mock(return_value=settings))
                with patch.object(module, inner_name, return_value=inner) as factory:
                    adapter = getattr(module, class_name)(model=model, **({} if module_name=='ollama_client' else {'api_key':'fake'}))
                if model == 'o4-mini':
                    self.assertEqual(factory.call_args.kwargs['required_temperature'], 1.0)
                actual = adapter.complete('task', max_tokens=20)
                self.assertEqual(actual.request_settings, settings)
                self.assertEqual(adapter.request_settings(20, 0), settings)
                self.assertEqual(actual.billing_tokens, response.billing_tokens)
                self.assertEqual(actual.usage_attempts, response.usage_attempts)
                self.assertTrue(actual.usage_complete)
                self.assertEqual(actual.finish_reason, 'max_output_tokens')

    def test_fallback_judge_policy_matches_harness_factory(self):
        from compass_experiments.config import EvalConfig
        from compass_experiments.models import create_client, provider_request_settings
        from unittest.mock import MagicMock
        config = EvalConfig(api_key="fake")
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            for model in ["o3-mini", "chatgpt-4o-latest", "o4-mini", "gpt-5-mini"]:
                with self.subTest(model=model):
                    client = create_client(config, model)
                    self.assertEqual(provider_request_settings(model, 180, 0),
                                     client.request_settings(180, 0))
