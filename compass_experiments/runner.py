import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple

from tqdm import tqdm

from compass_experiments.config import DEFAULT_SCENARIO_CONFIGS, EvalConfig, model_config, model_provider
from compass_experiments.metrics.accuracy import extract_mc_answer, multiple_choice_accuracy
from compass_experiments.metrics.calibration import compute_ece, extract_choice_probs
from compass_experiments.metrics.builtin_tic_suites import STYLE_SUITE_REGISTRY
from compass_experiments.metrics.pairwise_preference import analyze_pairwise_preference
from compass_experiments.metrics.persona_transfer import analyze_persona_transfer
from compass_experiments.metrics.quirk_detection import QUIRK_PROMPTS, SYSTEM_PROMPTS, find_quirks
from compass_experiments.metrics.linkedin_style import (
    LINKEDIN_SUITE,
    analyze_outputs,
    compute_uplift,
)
from compass_experiments.metrics.robustness import inject_typos
from compass.evaluation.contracts import detector_definition, fingerprint, validate_suite_identities
from compass.judges.context import PROMPT_VERSION as JUDGE_PROMPT_VERSION
from compass.judges.parsing import SCORING_VERSION

from compass_experiments.metrics.tic_suite import (
    LLMJudgeDetector,
    LLM_JUDGE_SYSTEM_PROMPT,
    TicSuite,
    analyze_text,
    suite_uses_llm_judge,
    summarize_detector_records,
    summarize_outputs,
)
from compass_experiments.models import CompletionClient, create_client
from compass_experiments.runner_support import (
    TIC_CHECKPOINT_SCHEMA_VERSION,
    ScenarioResults,
    aggregate_scenario_results,
    extract_judge_disagreements,
    load_scenario_checkpoint,
    load_tic_checkpoint,
    record_task_type,
    summarize_judge_disagreements,
)
from compass_experiments.scenarios.base import Instance, Scenario
from compass_experiments.scenarios.bbq import BBQScenario
from compass_experiments.scenarios.boolq import BoolQScenario
from compass_experiments.scenarios.hellaswag import HellaSwagScenario
from compass_experiments.scenarios.mmlu import MMLUScenario
from compass_experiments.scenarios.truthfulqa import TruthfulQAScenario

logger = logging.getLogger(__name__)
TicAnalyzeFn = Callable[[Dict[str, List[str]]], Dict[str, dict]]
TicUpliftFn = Callable[[Dict[str, dict], str], Dict[str, dict]]

SCENARIO_REGISTRY: Dict[str, type] = {
    "mmlu":       MMLUScenario,
    "truthfulqa": TruthfulQAScenario,
    "hellaswag":  HellaSwagScenario,
    "boolq":      BoolQScenario,
    "bbq":        BBQScenario,
}


def estimate_model_cost(model_name: str, prompts: List[str], max_tokens: int = 1) -> float:
    cfg = model_config(model_name)
    # GPT-5 reasoning models use 10x token budget for reasoning + output
    actual_max_tokens = max_tokens * 10 if model_name.startswith("gpt-5") else max_tokens
    n_input = sum(len(p) // 4 for p in prompts)
    n_output = len(prompts) * actual_max_tokens
    return (
        n_input * cfg.input_cost_per_million / 1_000_000
        + n_output * cfg.output_cost_per_million / 1_000_000
    )


@dataclass
class EvalResults:
    model:            str
    scenario_results: Dict[str, ScenarioResults] = field(default_factory=dict)
    quirk_results:    Optional[List[dict]]        = None
    linkedin_results: Optional[dict]              = None
    tic_results:      Dict[str, dict]             = field(default_factory=dict)
    persona_transfer_results: Dict[str, dict]     = field(default_factory=dict)
    pairwise_preference_results: Dict[str, dict]  = field(default_factory=dict)
    token_usage:      dict                        = field(default_factory=dict)
    total_cost_usd:   float                       = 0.0


class EvalRunner:
    def __init__(self, config: EvalConfig):
        self.config = config

    # ── public API ───────────────────────────────────────────────────────────

    def estimate_cost(self) -> Dict[str, float]:
        estimates = {}
        for model_name in self.config.models:
            total  = 0.0
            for sname in self.config.scenarios:
                cfg = DEFAULT_SCENARIO_CONFIGS[sname]
                n   = self.config.n_samples or cfg.n_samples
                dummy = ["x" * 1600] * n
                base  = estimate_model_cost(model_name, dummy, max_tokens=1)
                total += base * (2 if self.config.run_robustness else 1)
            if self.config.run_quirk_detection:
                n_quirk = len(QUIRK_PROMPTS) * len(SYSTEM_PROMPTS)
                total  += estimate_model_cost(model_name, ["x" * 400] * n_quirk, max_tokens=150)
            for _suite_key, suite, samples in self._enabled_tic_suites():
                n_calls = len(suite.prompts) * len(suite.conditions) * samples
                total += estimate_model_cost(
                    model_name,
                    ["x" * 100] * n_calls,
                    max_tokens=suite.max_tokens,
                )
                if self.config.judge_model and suite_uses_llm_judge(suite):
                    judge_detectors = [
                        detector for detector in suite.detectors
                        if isinstance(detector, LLMJudgeDetector)
                    ]
                    judge_calls = n_calls * len(judge_detectors)
                    max_judge_tokens = max(
                        (detector.max_tokens for detector in judge_detectors),
                        default=1,
                    )
                    total += estimate_model_cost(
                        self.config.judge_model,
                        ["x" * 1200] * judge_calls,
                        max_tokens=max_judge_tokens,
                    )
                    if self.config.judge_model_secondary:
                        total += estimate_model_cost(
                            self.config.judge_model_secondary,
                            ["x" * 1200] * judge_calls,
                            max_tokens=max_judge_tokens,
                        )
            estimates[model_name] = round(total, 4)
        return estimates

    def run(self) -> Dict[str, EvalResults]:
        out = {}
        for model_name in self.config.models:
            logger.info("Evaluating: %s", model_name)
            client = create_client(self.config, model_name)
            out[model_name] = self._run_model(client, model_name)
        if self.config.run_pairwise_preference:
            pairwise = analyze_pairwise_preference(
                out,
                detector_name=self.config.pairwise_detector,
                method=self.config.pairwise_method,
                segment_by=self.config.pairwise_segment_by,
                time_decay_halflife_days=self.config.pairwise_time_decay_halflife_days,
            )
            for result in out.values():
                result.pairwise_preference_results = pairwise
        return out

    # ── checkpointing ────────────────────────────────────────────────────────

    def _checkpoint_path(self, model_name: str, scenario_name: str) -> str:
        safe = model_name.replace("/", "_").replace(":", "_")
        ckpt_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        return os.path.join(ckpt_dir, f"{safe}__{scenario_name}.jsonl")

    @staticmethod
    def _load_checkpoint(path: str) -> Dict[str, dict]:
        return load_scenario_checkpoint(path, logger)

    # ── internal ─────────────────────────────────────────────────────────────

    def _run_model(self, client: CompletionClient, model_name: str) -> EvalResults:
        result = EvalResults(model=model_name)

        if not self.config.linkedin_only and not self.config.style_only:
            for sname in self.config.scenarios:
                if sname not in SCENARIO_REGISTRY:
                    logger.warning("Unknown scenario: %s — skipping", sname)
                    continue
                cfg = DEFAULT_SCENARIO_CONFIGS[sname]
                n   = self.config.n_samples or cfg.n_samples
                logger.info("  Scenario: %s  (n=%d)", sname, n)

                scenario: Scenario = SCENARIO_REGISTRY[sname]()
                try:
                    test_instances, few_shot = scenario.load(n, cfg.n_few_shot, self.config.seed)
                except Exception as exc:
                    logger.error("  Failed to load %s: %s", sname, exc)
                    continue

                ckpt_path = self._checkpoint_path(model_name, sname)
                result.scenario_results[sname] = self._eval_scenario(
                    client, scenario, test_instances, few_shot,
                    run_robustness=self.config.run_robustness,
                    checkpoint_path=ckpt_path,
                    model_name=model_name,
                )

        if self.config.run_quirk_detection:
            logger.info("  Quirk detection...")
            result.quirk_results = self._run_quirk_detection(client)

        # One-off safety-off Gemini client, created lazily and reused if self_harm_safety
        # appears in the suite list. Kept separate so its tokens can be merged below.
        safety_client: Optional[CompletionClient] = None

        for suite_key, suite, samples in self._enabled_tic_suites():
            logger.info("  Style suite: %s", suite_key)
            # SELF_HARM_SAFETY_SUITE sends adversarial prompts that Gemini's default
            # content filters would block. Use a safety-off client so the model's own
            # refusal behavior is measured, not the provider's pre-filter.
            suite_client = client
            if suite_key == "self_harm_safety" and model_provider(model_name) == "google":
                if safety_client is None:
                    safety_client = create_client(
                        self.config, model_name, disable_safety_filters=True
                    )
                suite_client = safety_client
            if suite_key == "linkedin":
                suite_result = self._run_linkedin_style(suite_client, model_name, samples=samples)
                result.linkedin_results = suite_result
            else:
                suite_result = self._run_tic_suite(
                    client=suite_client,
                    model_name=model_name,
                    suite=suite,
                    samples=samples,
                    progress_label=suite_key,
                )
            result.tic_results[suite.name] = suite_result

        if self.config.run_persona_transfer and result.tic_results:
            suites_by_name = {
                suite.name: suite
                for _suite_key, suite, _samples in self._enabled_tic_suites()
            }
            result.persona_transfer_results = analyze_persona_transfer(
                result.tic_results,
                suites_by_name,
            )

        # Merge tokens and cost from the safety-off client (if used) into the totals.
        tokens = dict(client.total_tokens)
        cost = client.total_cost_usd
        if safety_client is not None:
            for k, v in safety_client.total_tokens.items():
                tokens[k] = tokens.get(k, 0) + v
            cost += safety_client.total_cost_usd
        result.token_usage    = tokens
        result.total_cost_usd = cost
        return result

    def _eval_scenario(
        self,
        client:           CompletionClient,
        scenario:         Scenario,
        instances:        List[Instance],
        few_shot:         List[Instance],
        run_robustness:   bool,
        checkpoint_path:  str,
        model_name: Optional[str] = None,
    ) -> ScenarioResults:
        # Prepare the full request set before reading or appending a checkpoint.
        # Formatting errors are configuration errors, not failed model responses.
        if len({inst.id for inst in instances}) != len(instances):
            raise ValueError("Scenario instance IDs must be unique")
        requests = {}
        contracts = {}
        request_settings = {"max_tokens": 1, "logprobs": True, "top_logprobs": 20,
                            "temperature": 0.0, "system": "You are a helpful assistant."}
        perturbed_settings = {"max_tokens": 1, "logprobs": False,
                              "temperature": 0.0, "system": "You are a helpful assistant."}
        for inst in instances:
            prompt = scenario.format_prompt(inst, few_shot)
            p_prompt = None
            if run_robustness:
                p_inst = Instance(
                    id=inst.id + "_pert",
                    input=inject_typos(inst.input, p=0.05, seed=self.config.seed),
                    choices=inst.choices, choice_texts=inst.choice_texts,
                    correct=inst.correct, metadata=inst.metadata,
                )
                p_prompt = scenario.format_prompt(p_inst, few_shot)
            requests[inst.id] = (prompt, p_prompt)
            contracts[inst.id] = fingerprint({
                "version": 1, "scenario": scenario.name,
                "model": model_name or getattr(client, "model", self.config.models),
                "client": f"{type(client).__module__}.{type(client).__qualname__}",
                "instance": asdict(inst), "prompt": prompt, "perturbed_prompt": p_prompt,
                "seed": self.config.seed, "run_robustness": run_robustness,
                "request": request_settings,
                "perturbed_request": perturbed_settings if run_robustness else None,
            })
        done = self._load_checkpoint(checkpoint_path)
        manifest_path = checkpoint_path + ".manifest.json"
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                previous = json.load(f)
            if any(contracts.get(key) != value for key, value in previous.items()):
                raise ValueError("Scenario checkpoint inputs changed; use a fresh output directory")
        elif done:
            raise ValueError("Scenario checkpoint inputs changed or lack a verified manifest; use a fresh output directory")
        for key, record in done.items():
            if key not in contracts or record.get("scenario_contract") != contracts[key]:
                raise ValueError("Scenario checkpoint inputs changed or lack a verified contract; use a fresh output directory")

        temporary = manifest_path + ".tmp"
        with open(temporary, "w") as f:
            json.dump(contracts, f, sort_keys=True)
        os.replace(temporary, manifest_path)
        failures = 0
        with open(checkpoint_path, "a") as ckpt_f:
            for inst in tqdm(instances, desc=scenario.name, leave=False):
                if inst.id in done:
                    continue

                try:
                    prompt, p_prompt = requests[inst.id]
                    resp   = client.complete(prompt, **request_settings)
                    pred   = extract_mc_answer(resp.completion, inst.choices)
                    probs, coverage = extract_choice_probs(resp.logprobs, inst.choices)
                    conf   = probs.get(pred, 1.0 / len(inst.choices))
                    ok     = pred == inst.correct

                    p_ok: Optional[bool] = None
                    if run_robustness:
                        p_resp   = client.complete(p_prompt, **perturbed_settings)
                        p_pred   = extract_mc_answer(p_resp.completion, inst.choices)
                        p_ok     = p_pred == inst.correct

                    record: dict = {
                        "id":                    inst.id,
                        "scenario_contract":     contracts[inst.id],
                        "created_at":            datetime.now(timezone.utc).isoformat(),
                        "prompt":                prompt,
                        "completion":            resp.completion,
                        "prediction":            pred,
                        "correct":               inst.correct,
                        "is_correct":            ok,
                        "confidence":            conf,
                        "coverage":              coverage,
                        "perturbed_is_correct":  p_ok,
                        "metadata":              inst.metadata,
                    }
                    if self.config.run_tag:
                        record["run_tag"] = self.config.run_tag
                    done[inst.id] = record
                    ckpt_f.write(json.dumps(record) + "\n")
                    ckpt_f.flush()

                except Exception as exc:
                    failures += 1
                    logger.error("  Instance %s failed: %s", inst.id, exc)

        if failures:
            raise RuntimeError(f"{failures} scenario instances failed; resume to retry missing results")
        return self._aggregate(scenario.name, list(done.values()), len(instances), checkpoint_path)

    @staticmethod
    def _aggregate(
        name: str,
        records: List[dict],
        n_requested: int,
        checkpoint_path: str,
    ) -> ScenarioResults:
        return aggregate_scenario_results(
            name=name,
            records=records,
            n_requested=n_requested,
            checkpoint_path=checkpoint_path,
            logger=logger,
        )

    def _run_quirk_detection(self, client: CompletionClient) -> List[dict]:
        condition_texts: Dict[str, List[str]] = {c: [] for c in SYSTEM_PROMPTS}
        for cond, sys_prompt in SYSTEM_PROMPTS.items():
            for q in tqdm(QUIRK_PROMPTS, desc=f"quirks/{cond}", leave=False):
                try:
                    resp = client.complete(q, max_tokens=120, logprobs=False, system=sys_prompt)
                    condition_texts[cond].append(resp.completion)
                except Exception as exc:
                    logger.error("  Quirk prompt failed (%s/%s): %s", cond, q[:30], exc)
        return find_quirks(condition_texts)

    def _enabled_tic_suites(self) -> List[Tuple[str, TicSuite, int]]:
        enabled: List[Tuple[str, TicSuite, int]] = []
        seen = set()
        if self.config.run_linkedin_style:
            enabled.append(("linkedin", LINKEDIN_SUITE, self.config.linkedin_samples))
            seen.add("linkedin")

        for suite_key in self.config.style_suites:
            if suite_key in seen:
                continue
            suite = STYLE_SUITE_REGISTRY.get(suite_key)
            if not suite:
                raise ValueError(f"Unknown style suite: {suite_key}")
            enabled.append((suite_key, suite, self.config.style_samples))
            seen.add(suite_key)
        return enabled

    def _run_linkedin_style(
        self,
        client: CompletionClient,
        model_name: str,
        samples: Optional[int] = None,
    ) -> dict:
        return self._run_tic_suite(
            client=client,
            model_name=model_name,
            suite=LINKEDIN_SUITE,
            samples=samples or self.config.linkedin_samples,
            progress_label="linkedin-style",
            analyze_fn=analyze_outputs,
            uplift_fn=compute_uplift,
        )

    def _run_tic_suite(
        self,
        client: CompletionClient,
        model_name: str,
        suite: TicSuite,
        samples: int,
        progress_label: Optional[str] = None,
        analyze_fn: Optional[TicAnalyzeFn] = None,
        uplift_fn: Optional[TicUpliftFn] = None,
    ) -> dict:
        condition_outputs: Dict[str, List[str]] = {c.name: [] for c in suite.conditions}
        validate_suite_identities(suite)
        # Group completions by task type, then by condition, so reports can show
        # task-level slices without re-running any model calls.
        task_outputs: Dict[str, Dict[str, List[str]]] = {
            prompt.task_type: {c.name: [] for c in suite.conditions}
            for prompt in suite.prompts
        }
        condition_detector_records: Dict[str, List[Dict[str, dict]]] = {
            c.name: [] for c in suite.conditions
        }
        task_detector_records: Dict[str, Dict[str, List[Dict[str, dict]]]] = {
            prompt.task_type: {c.name: [] for c in suite.conditions}
            for prompt in suite.prompts
        }
        judge_disagreement_rows: List[Dict[str, Any]] = []
        safe_model = model_name.replace("/", "_").replace(":", "_")
        ckpt_path = os.path.join(
            self.config.output_dir,
            "checkpoints",
            f"{safe_model}__{suite.checkpoint_stem}.jsonl",
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        run_contract = fingerprint({
            "version": 1, "model": model_name, "suite": suite.name,
            "prompts": [vars(p) for p in suite.prompts],
            "conditions": [vars(c) for c in suite.conditions],
            "max_tokens": suite.max_tokens, "requested_temperature": 0.0,
            "detectors": [detector_definition(d) for d in suite.detectors],
            "judge_model": self.config.judge_model,
            "judge_model_secondary": self.config.judge_model_secondary,
            "judge_system_prompt": LLM_JUDGE_SYSTEM_PROMPT,
            "judge_context_mode": "user",
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "scoring_version": SCORING_VERSION,
        })
        generations_path = ckpt_path + ".generations.jsonl"
        requests_path = ckpt_path + ".requests.jsonl"
        saved_generations = {}
        requested_samples = set()
        for artifact in (ckpt_path, generations_path, requests_path):
            if os.path.exists(artifact):
                with open(artifact) as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row.get("run_contract") != run_contract:
                            raise ValueError(
                                "Style checkpoint inputs changed or lack a verified run contract. "
                                "Use a fresh --output-dir; --run-tag does not isolate checkpoints."
                            )
                        if artifact == generations_path:
                            key = (row["condition"], row["prompt_idx"], row["sample_idx"])
                            saved_generations[key] = row
                        elif artifact == requests_path:
                            requested_samples.add((row["condition"], row["prompt_idx"], row["sample_idx"]))

        judge_client: Optional[CompletionClient] = None
        judge_client_secondary: Optional[CompletionClient] = None
        judge_fn = None
        judge_fn_secondary = None
        if suite_uses_llm_judge(suite):
            if not self.config.judge_model:
                raise ValueError(f"Suite {suite.name!r} requires --judge-model.")
            judge_client = create_client(self.config, self.config.judge_model)

            def judge_fn(prompt: str, max_tokens: int) -> str:
                resp = judge_client.complete(
                    prompt,
                    max_tokens=max_tokens,
                    logprobs=False,
                    system=LLM_JUDGE_SYSTEM_PROMPT,
                )
                return resp.completion

            if self.config.judge_model_secondary:
                judge_client_secondary = create_client(
                    self.config,
                    self.config.judge_model_secondary,
                )

                def judge_fn_secondary(prompt: str, max_tokens: int) -> str:
                    resp = judge_client_secondary.complete(
                        prompt,
                        max_tokens=max_tokens,
                        logprobs=False,
                        system=LLM_JUDGE_SYSTEM_PROMPT,
                    )
                    return resp.completion

        done, resumed_outputs = self._load_tic_checkpoint(ckpt_path, suite)
        for condition, outputs in resumed_outputs.items():
            condition_outputs[condition].extend(outputs)
        for rec in done.values():
            condition = rec.get("condition")
            task_type = self._record_task_type(rec, suite)
            completion = rec.get("completion")
            detectors = rec.get("detectors")
            if condition in condition_outputs and completion and task_type in task_outputs:
                task_outputs[task_type][condition].append(completion)
            if condition in condition_detector_records and isinstance(detectors, dict):
                condition_detector_records[condition].append(detectors)
                if task_type in task_detector_records:
                    task_detector_records[task_type][condition].append(detectors)
            judge_disagreement_rows.extend(self._extract_judge_disagreements(rec))

        existing_sample_indices: DefaultDict[Tuple[str, int], List[int]] = defaultdict(list)
        pending_samples = (requested_samples | set(saved_generations)) - set(done)
        if self.config.style_samples_increment:
            for condition_name, prompt_idx, sample_idx in done.keys():
                # Defensive guard: checkpoints are expected to store numeric sample_idx,
                # but old or hand-edited files might not.
                if isinstance(sample_idx, int):
                    existing_sample_indices[(condition_name, prompt_idx)].append(sample_idx)
            logger.info(
                "    Incremental style sampling enabled: appending %d samples per prompt/condition",
                samples,
            )

        planned = {}
        with open(requests_path, "a") as request_file:
            for condition in suite.conditions:
                for prompt_idx, prompt in enumerate(suite.prompts):
                    slot = (condition.name, prompt_idx)
                    pending = {key[2] for key in pending_samples if key[:2] == slot}
                    if self.config.style_samples_increment:
                        existing = existing_sample_indices.get(slot, [])
                        start_idx = max(existing) + 1 if existing else 0
                        indices = sorted(pending) if pending_samples else range(start_idx, start_idx + samples)
                    else:
                        indices = sorted(set(range(samples)) | pending)
                    planned[slot] = list(indices)
                    for sample_idx in indices:
                        key = (*slot, sample_idx)
                        if key not in requested_samples:
                            request_file.write(json.dumps({
                                "run_contract": run_contract, "condition": condition.name,
                                "prompt_idx": prompt_idx, "sample_idx": sample_idx,
                            }) + "\n")
        total_calls = sum(len(indices) for indices in planned.values())
        failures = 0
        pbar = tqdm(total=total_calls, desc=progress_label or suite.name, leave=False)

        with open(ckpt_path, "a") as ckpt_f, open(generations_path, "a") as gen_f:
            for condition in suite.conditions:
                for prompt_idx, prompt in enumerate(suite.prompts):
                    sample_indices = planned[(condition.name, prompt_idx)]

                    for sample_idx in sample_indices:
                        pbar.update(1)
                        key = (condition.name, prompt_idx, sample_idx)
                        if key in done:
                            continue

                        try:
                            if key in saved_generations:
                                generation = saved_generations[key]
                                completion = generation["completion"]
                            else:
                                resp = client.complete(
                                    prompt.text,
                                    max_tokens=suite.max_tokens,
                                    logprobs=False,
                                    system=condition.system_prompt,
                                )
                                completion = resp.completion
                                generation = {
                                    "run_contract": run_contract, "condition": condition.name,
                                    "prompt_idx": prompt_idx, "sample_idx": sample_idx,
                                    "prompt": prompt.text, "system_prompt": condition.system_prompt,
                                    "completion": completion,
                                }
                                gen_f.write(json.dumps(generation) + "\n")
                                gen_f.flush()
                                saved_generations[key] = generation
                            detector_results = analyze_text(
                                completion,
                                suite,
                                judge_fn=judge_fn,
                                context=generation["prompt"],
                            )
                            secondary_results = None
                            if judge_fn_secondary is not None:
                                secondary_results = analyze_text(
                                    completion,
                                    suite,
                                    judge_fn=judge_fn_secondary,
                                    context=generation["prompt"],
                                )
                                self._annotate_secondary_judge_metadata(
                                    detector_results=detector_results,
                                    secondary_results=secondary_results,
                                )
                            detector_record = {
                                name: result.to_dict()
                                for name, result in detector_results.items()
                            }
                            condition_detector_records[condition.name].append(detector_record)
                            task_detector_records[prompt.task_type][condition.name].append(detector_record)

                            record = {
                                "run_contract": run_contract,
                                "schema_version": TIC_CHECKPOINT_SCHEMA_VERSION,
                                "created_at": datetime.now(timezone.utc).isoformat(),
                                "model": model_name,
                                "suite": suite.name,
                                "condition": condition.name,
                                "prompt_id": prompt.id,
                                "task_type": prompt.task_type,
                                "prompt_idx": prompt_idx,
                                "sample_idx": sample_idx,
                                "prompt": prompt.text,
                                "completion": completion,
                                "detectors": detector_record,
                            }
                            if self.config.run_tag:
                                record["run_tag"] = self.config.run_tag
                            record.update(suite.extra_record_fields(completion, detector_results))
                            ckpt_f.write(json.dumps(record) + "\n")
                            ckpt_f.flush()
                            condition_outputs[condition.name].append(completion)
                            task_outputs[prompt.task_type][condition.name].append(completion)
                            judge_disagreement_rows.extend(
                                self._extract_judge_disagreements(record)
                            )
                        except Exception as exc:
                            failures += 1
                            logger.error(
                                "  %s prompt failed (%s/%s): %s",
                                suite.name,
                                condition.name,
                                prompt.text[:30],
                                exc,
                            )
        pbar.close()

        if failures:
            raise RuntimeError(
                f"Suite {suite.name!r} had {failures} failed samples. "
                "Successful checkpoints were saved; retry before reporting."
            )

        uses_judge = suite_uses_llm_judge(suite)
        if analyze_fn:
            analysis = analyze_fn(condition_outputs)
        elif uses_judge:
            analysis = summarize_detector_records(condition_detector_records, suite)
        else:
            analysis = summarize_outputs(condition_outputs, suite)
        uplift = (
            uplift_fn(analysis, suite.baseline_condition)
            if uplift_fn
            else {}
        )
        result = {
            "suite": suite.name,
            "analysis": analysis,
            "uplift": uplift,
            "task_segments": {
                task_type: (
                    analyze_fn(outputs)
                    if analyze_fn
                    else summarize_detector_records(task_detector_records[task_type], suite)
                    if uses_judge
                    else summarize_outputs(outputs, suite)
                )
                for task_type, outputs in task_outputs.items()
            },
            "checkpoint_path": ckpt_path,
        }
        if judge_client and self.config.judge_model:
            result["judge"] = {
                "model": self.config.judge_model,
                "token_usage": judge_client.total_tokens,
                "cost_usd": round(judge_client.total_cost_usd, 6),
            }
        if judge_client_secondary and self.config.judge_model_secondary:
            result["judge_secondary"] = {
                "model": self.config.judge_model_secondary,
                "token_usage": judge_client_secondary.total_tokens,
                "cost_usd": round(judge_client_secondary.total_cost_usd, 6),
            }
            result["judge_agreement"] = self._summarize_judge_disagreements(
                judge_disagreement_rows,
                threshold=self.config.judge_disagreement_threshold,
            )
        return result

    @staticmethod
    def _annotate_secondary_judge_metadata(
        detector_results: Dict[str, Any],
        secondary_results: Dict[str, Any],
    ) -> None:
        for detector_name, primary in detector_results.items():
            secondary = secondary_results.get(detector_name)
            if secondary is None:
                continue

            delta = abs(primary.score - secondary.score)
            primary.metadata.update(
                {
                    "secondary_score": round(secondary.score, 4),
                    "secondary_hit": bool(secondary.hit),
                    "secondary_confidence": secondary.metadata.get("confidence"),
                    "secondary_rationale": str(secondary.metadata.get("rationale", ""))[:500],
                    "score_delta": round(delta, 4),
                    "agreement": bool(primary.hit == secondary.hit),
                }
            )

    @staticmethod
    def _extract_judge_disagreements(record: Dict[str, Any]) -> List[Dict[str, Any]]:
        return extract_judge_disagreements(record)

    @staticmethod
    def _summarize_judge_disagreements(
        rows: List[Dict[str, Any]],
        threshold: float,
    ) -> Dict[str, Any]:
        return summarize_judge_disagreements(rows, threshold)

    @staticmethod
    def _record_task_type(rec: dict, suite: TicSuite) -> str:
        return record_task_type(rec, suite)

    @staticmethod
    def _load_tic_checkpoint(
        path: str,
        suite: TicSuite,
    ) -> Tuple[Dict[tuple, dict], Dict[str, List[str]]]:
        return load_tic_checkpoint(path, suite, logger)
