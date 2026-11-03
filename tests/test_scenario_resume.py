from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from compass_experiments.runner import EvalRunner
from compass_experiments.scenarios.base import Instance
from tests._test_base import make_eval_config


class Scenario:
    name = 'test'

    def format_prompt(self, inst, few_shot):
        return str([(item.input, item.correct) for item in few_shot]) + inst.input


def instance():
    return Instance('id', 'Question', ['A', 'B'], ['yes', 'no'], 'A')


def run(tmp_path, client, instances=None, few_shot=None, robustness=False, seed=42):
    return EvalRunner(make_eval_config(output_dir=str(tmp_path), seed=seed))._eval_scenario(
        client, Scenario(), instances if instances is not None else [instance()],
        few_shot or [], robustness, str(tmp_path/'scenario.jsonl'), model_name='test-model',
    )


def client():
    return Mock(complete=Mock(return_value=SimpleNamespace(completion='A', logprobs=[])))


def test_resume_makes_no_calls(tmp_path):
    fake = client()
    run(tmp_path, fake)
    run(tmp_path, fake)
    assert fake.complete.call_count == 1


@pytest.mark.parametrize('change', ['prompt', 'answer', 'few_shot', 'robustness', 'seed', 'subset'])
def test_changed_inputs_fail_before_calls_or_writes(tmp_path, change):
    run(tmp_path, client())
    before = (tmp_path/'scenario.jsonl').read_bytes()
    args = {
        'prompt': {'instances': [replace(instance(), input='Changed')]},
        'answer': {'instances': [replace(instance(), correct='B')]},
        'few_shot': {'few_shot': [instance()]},
        'robustness': {'robustness': True},
        'seed': {'seed': 7},
        'subset': {'instances': []},
    }[change]
    fake = client()
    with pytest.raises(ValueError, match='inputs changed'):
        run(tmp_path, fake, **args)
    fake.complete.assert_not_called()
    assert (tmp_path/'scenario.jsonl').read_bytes() == before


@pytest.mark.parametrize('contents', ['{bad json', '{"id":"id"}\n', '{"id":"id"}\n{"id":"id"}\n'])
def test_invalid_or_legacy_checkpoint_is_preserved(tmp_path, contents):
    path = tmp_path/'scenario.jsonl'
    path.write_text(contents)
    fake = client()
    with pytest.raises(ValueError):
        run(tmp_path, fake)
    fake.complete.assert_not_called()
    assert path.read_text() == contents


def test_duplicate_input_ids_rejected_before_calls(tmp_path):
    fake = client()
    with pytest.raises(ValueError, match='must be unique'):
        run(tmp_path, fake, instances=[instance(), instance()])
    fake.complete.assert_not_called()


def test_failed_instances_remain_in_plan_and_resume(tmp_path):
    fake = client()
    fake.complete.side_effect = [fake.complete.return_value, RuntimeError('offline')]
    instances = [instance(), replace(instance(), id='second')]
    with pytest.raises(RuntimeError, match='resume to retry'):
        run(tmp_path, fake, instances=instances)
    with pytest.raises(ValueError, match='inputs changed'):
        run(tmp_path, client())  # Cannot silently drop the failed planned instance.
    fake = client()
    result = run(tmp_path, fake, instances=instances)
    assert fake.complete.call_count == 1
    assert result.n_completed == result.n_requested == 2


def test_request_settings_are_sent_to_both_calls(tmp_path):
    fake = client()
    run(tmp_path, fake, robustness=True)
    assert fake.complete.call_count == 2
    for call in fake.complete.call_args_list:
        assert call.kwargs['temperature'] == 0.0
        assert call.kwargs['system'] == 'You are a helpful assistant.'
        assert call.kwargs['max_tokens'] == 1
    assert fake.complete.call_args_list[0].kwargs['logprobs'] is True
    assert fake.complete.call_args_list[1].kwargs['logprobs'] is False
