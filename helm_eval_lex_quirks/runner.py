import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple

from tqdm import tqdm

from helm_eval_lex_quirks.config import DEFAULT_SCENARIO_CONFIGS, EvalConfig, model_config
from helm_eval_lex_quirks.metrics.accuracy import extract_mc_answer, multiple_choice_accuracy
from helm_eval_lex_quirks.metrics.calibration import compute_ece, extract_choice_probs
from helm_eval_lex_quirks.metrics.builtin_tic_suites import STYLE_SUITE_REGISTRY
from helm_eval_lex_quirks.metrics.pairwise_preference import analyze_pairwise_preference
from helm_eval_lex_quirks.metrics.persona_transfer import analyze_persona_transfer
from helm_eval_lex_quirks.metrics.quirk_detection import QUIRK_PROMPTS, SYSTEM_PROMPTS, find_quirks
from helm_eval_lex_quirks.metrics.linkedin_style import (
    LINKEDIN_SUITE,
    analyze_outputs,
    compute_uplift,
)
from helm_eval_lex_quirks.metrics.robustness import compute_robustness_score, inject_typos
from helm_eval_lex_quirks.metrics.tic_suite import (
    LLMJudgeDetector,
    LLM_JUDGE_SYSTEM_PROMPT,
    TicSuite,
    analyze_text,
    suite_uses_llm_judge,
    summarize_detector_records,
    summarize_outputs,
)
from helm_eval_lex_quirks.models import CompletionClient, create_client
from helm_eval_lex_quirks.scenarios.base import Instance, Scenario
from helm_eval_lex_quirks.scenarios.bbq import BBQScenario
from helm_eval_lex_quirks.scenarios.boolq import BoolQScenario
from helm_eval_lex_quirks.scenarios.hellaswag import HellaSwagScenario
from helm_eval_lex_quirks.scenarios.mmlu import MMLUScenario
from helm_eval_lex_quirks.scenarios.truthfulqa import TruthfulQAScenario

logger = logging.getLogger(__name__)
TIC_CHECKPOINT_SCHEMA_VERSION = 1
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
class ScenarioResults:
    name:             str
    accuracy:         float
    ece:              Optional[float]   # None when coverage was too low for all instances
    ece_coverage:     float             # fraction of instances with full logprob coverage
    robustness_score: Optional[float]
    n_completed:      int               # instances successfully evaluated
    n_requested:      int
    checkpoint_path:  str              # path to per-instance JSONL

    # BBQ-specific
    ambig_unknown_rate: Optional[float] = None
    disambig_accuracy:  Optional[float] = None


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
        done: Dict[str, dict] = {}
        if not os.path.exists(path):
            return done
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[rec["id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    pass  # corrupted line; skip
        if done:
            logger.info("    Resumed %d checkpointed instances from %s", len(done), path)
        return done

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
                )

        if self.config.run_quirk_detection:
            logger.info("  Quirk detection...")
            result.quirk_results = self._run_quirk_detection(client)

        for suite_key, suite, samples in self._enabled_tic_suites():
            logger.info("  Style suite: %s", suite_key)
            if suite_key == "linkedin":
                suite_result = self._run_linkedin_style(client, model_name, samples=samples)
                result.linkedin_results = suite_result
            else:
                suite_result = self._run_tic_suite(
                    client=client,
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

        result.token_usage    = client.total_tokens
        result.total_cost_usd = client.total_cost_usd
        return result

    def _eval_scenario(
        self,
        client:           CompletionClient,
        scenario:         Scenario,
        instances:        List[Instance],
        few_shot:         List[Instance],
        run_robustness:   bool,
        checkpoint_path:  str,
    ) -> ScenarioResults:
        done = self._load_checkpoint(checkpoint_path)

        with open(checkpoint_path, "a") as ckpt_f:
            for inst in tqdm(instances, desc=scenario.name, leave=False):
                if inst.id in done:
                    continue

                try:
                    prompt = scenario.format_prompt(inst, few_shot)
                    resp   = client.complete(prompt, max_tokens=1, logprobs=True, top_logprobs=20)
                    pred   = extract_mc_answer(resp.completion, inst.choices)
                    probs, coverage = extract_choice_probs(resp.logprobs, inst.choices)
                    conf   = probs.get(pred, 1.0 / len(inst.choices))
                    ok     = pred == inst.correct

                    p_ok: Optional[bool] = None
                    if run_robustness:
                        p_input  = inject_typos(inst.input, p=0.05, seed=self.config.seed)
                        p_inst   = Instance(
                            id=inst.id + "_pert",
                            input=p_input,
                            choices=inst.choices,
                            choice_texts=inst.choice_texts,
                            correct=inst.correct,
                            metadata=inst.metadata,
                        )
                        p_prompt = scenario.format_prompt(p_inst, few_shot)
                        p_resp   = client.complete(p_prompt, max_tokens=1, logprobs=False)
                        p_pred   = extract_mc_answer(p_resp.completion, inst.choices)
                        p_ok     = p_pred == inst.correct

                    record: dict = {
                        "id":                    inst.id,
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
                    logger.error("  Instance %s failed: %s — skipping", inst.id, exc)

        return self._aggregate(scenario.name, list(done.values()), len(instances), checkpoint_path)

    @staticmethod
    def _aggregate(
        name: str,
        records: List[dict],
        n_requested: int,
        checkpoint_path: str,
    ) -> ScenarioResults:
        if not records:
            return ScenarioResults(
                name=name, accuracy=0.0, ece=None, ece_coverage=0.0,
                robustness_score=None, n_completed=0, n_requested=n_requested,
                checkpoint_path=checkpoint_path,
            )

        predictions   = [r["prediction"]  for r in records]
        references    = [r["correct"]      for r in records]
        confidences   = [r["confidence"]   for r in records]
        correct_flags = [r["is_correct"]   for r in records]
        coverages     = [r.get("coverage", 1.0) for r in records]
        perturbed_ok  = [r["perturbed_is_correct"] for r in records
                         if r.get("perturbed_is_correct") is not None]

        accuracy = multiple_choice_accuracy(predictions, references)

        # Only include instances where all choices appeared in the logprobs
        full_cov_pairs = [(c, f) for c, f, cov in zip(confidences, correct_flags, coverages)
                          if cov >= 1.0]
        ece_coverage = len(full_cov_pairs) / len(records) if records else 0.0
        if full_cov_pairs:
            ece_confs, ece_flags = zip(*full_cov_pairs)
            ece = compute_ece(list(ece_confs), list(ece_flags))
        else:
            ece = None
            logger.warning(
                "%s: no instances had full logprob coverage — ECE cannot be computed", name
            )
        if 0 < ece_coverage < 1.0:
            n_low = len(records) - len(full_cov_pairs)
            logger.warning(
                "%s: %d/%d instances had incomplete logprob coverage and were excluded from ECE",
                name, n_low, len(records),
            )

        rob_score = None
        if perturbed_ok:
            perturbed_acc = sum(perturbed_ok) / len(perturbed_ok)
            rob_score     = compute_robustness_score(accuracy, perturbed_acc)

        sr = ScenarioResults(
            name=name,
            accuracy=round(accuracy, 4),
            ece=round(ece, 4) if ece is not None else None,
            ece_coverage=round(ece_coverage, 4),
            robustness_score=round(rob_score, 4) if rob_score is not None else None,
            n_completed=len(records),
            n_requested=n_requested,
            checkpoint_path=checkpoint_path,
        )

        # BBQ bias
        ambig    = [r for r in records if r.get("metadata", {}).get("context_condition") == "ambig"]
        disambig = [r for r in records if r.get("metadata", {}).get("context_condition") == "disambig"]
        if ambig:
            sr.ambig_unknown_rate = round(sum(r["is_correct"] for r in ambig) / len(ambig), 4)
        if disambig:
            sr.disambig_accuracy  = round(sum(r["is_correct"] for r in disambig) / len(disambig), 4)

        return sr

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

        total_calls = len(suite.prompts) * len(suite.conditions) * samples
        pbar = tqdm(total=total_calls, desc=progress_label or suite.name, leave=False)

        with open(ckpt_path, "a") as ckpt_f:
            for condition in suite.conditions:
                for prompt_idx, prompt in enumerate(suite.prompts):
                    for sample_idx in range(samples):
                        pbar.update(1)
                        key = (condition.name, prompt_idx, sample_idx)
                        if key in done:
                            continue

                        try:
                            resp = client.complete(
                                prompt.text,
                                max_tokens=suite.max_tokens,
                                logprobs=False,
                                system=condition.system_prompt,
                            )
                            completion = resp.completion
                            condition_outputs[condition.name].append(completion)
                            task_outputs[prompt.task_type][condition.name].append(completion)
                            detector_results = analyze_text(
                                completion,
                                suite,
                                judge_fn=judge_fn,
                            )
                            secondary_results = None
                            if judge_fn_secondary is not None:
                                secondary_results = analyze_text(
                                    completion,
                                    suite,
                                    judge_fn=judge_fn_secondary,
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
                            judge_disagreement_rows.extend(
                                self._extract_judge_disagreements(record)
                            )
                        except Exception as exc:
                            logger.error(
                                "  %s prompt failed (%s/%s): %s",
                                suite.name,
                                condition.name,
                                prompt.text[:30],
                                exc,
                            )
        pbar.close()

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
            if "confidence" not in primary.metadata:
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
        rows: List[Dict[str, Any]] = []
        detectors = record.get("detectors", {})
        if not isinstance(detectors, dict):
            return rows

        for detector_name, detector in detectors.items():
            if not isinstance(detector, dict):
                continue
            metadata = detector.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            if "secondary_score" not in metadata:
                continue

            rows.append(
                {
                    "condition": record.get("condition", "unknown"),
                    "task_type": record.get("task_type", "general"),
                    "prompt_id": record.get("prompt_id"),
                    "sample_idx": record.get("sample_idx"),
                    "detector": detector_name,
                    "primary_score": float(detector.get("score", 0.0)),
                    "secondary_score": float(metadata.get("secondary_score", 0.0)),
                    "score_delta": float(metadata.get("score_delta", 0.0)),
                    "agreement": bool(metadata.get("agreement", False)),
                }
            )
        return rows

    @staticmethod
    def _summarize_judge_disagreements(
        rows: List[Dict[str, Any]],
        threshold: float,
    ) -> Dict[str, Any]:
        def _init_bucket() -> Dict[str, float]:
            return {"n": 0.0, "agree": 0.0, "delta_sum": 0.0, "high_disagreement": 0.0}

        def _update(bucket: Dict[str, float], row: Dict[str, Any]) -> None:
            bucket["n"] += 1.0
            if row.get("agreement"):
                bucket["agree"] += 1.0
            delta = float(row.get("score_delta", 0.0))
            bucket["delta_sum"] += delta
            if delta >= threshold:
                bucket["high_disagreement"] += 1.0

        def _finalize(bucket: Dict[str, float]) -> Dict[str, float]:
            n = bucket["n"]
            if n <= 0:
                return {
                    "n": 0,
                    "agreement_pct": 0.0,
                    "mean_abs_score_delta": 0.0,
                    "high_disagreement_pct": 0.0,
                }
            return {
                "n": int(n),
                "agreement_pct": round(100.0 * bucket["agree"] / n, 1),
                "mean_abs_score_delta": round(bucket["delta_sum"] / n, 4),
                "high_disagreement_pct": round(100.0 * bucket["high_disagreement"] / n, 1),
            }

        overall = _init_bucket()
        by_condition: DefaultDict[str, Dict[str, float]] = defaultdict(_init_bucket)
        by_task_type: DefaultDict[str, Dict[str, float]] = defaultdict(_init_bucket)

        for row in rows:
            _update(overall, row)
            _update(by_condition[row.get("condition", "unknown")], row)
            _update(by_task_type[row.get("task_type", "general")], row)

        top_disagreements = sorted(
            rows,
            key=lambda row: row.get("score_delta", 0.0),
            reverse=True,
        )[:10]

        return {
            "threshold": threshold,
            "overall": _finalize(overall),
            "conditions": {
                key: _finalize(bucket)
                for key, bucket in sorted(by_condition.items())
            },
            "task_types": {
                key: _finalize(bucket)
                for key, bucket in sorted(by_task_type.items())
            },
            "top_disagreements": top_disagreements,
        }

    @staticmethod
    def _record_task_type(rec: dict, suite: TicSuite) -> str:
        task_type = rec.get("task_type")
        if task_type:
            return task_type

        prompt_id = rec.get("prompt_id")
        if prompt_id:
            for prompt in suite.prompts:
                if prompt.id == prompt_id:
                    return prompt.task_type

        prompt_idx = rec.get("prompt_idx")
        if isinstance(prompt_idx, int) and 0 <= prompt_idx < len(suite.prompts):
            return suite.prompts[prompt_idx].task_type

        # Legacy checkpoints may lack prompt metadata; fall back to the generic bucket.
        return "general"

    @staticmethod
    def _load_tic_checkpoint(
        path: str,
        suite: TicSuite,
    ) -> Tuple[Dict[tuple, dict], Dict[str, List[str]]]:
        """
        Load tic-suite checkpoint records.

        Schema v1 records include schema_version, suite, prompt_id, and
        normalized detectors. Records without schema_version are treated as
        legacy v0 LinkedIn records and are accepted if they contain the minimal
        resumption keys used by the original checkpoint format.
        """
        done: Dict[tuple, dict] = {}
        condition_outputs: Dict[str, List[str]] = {c.name: [] for c in suite.conditions}
        valid_conditions = set(condition_outputs)

        if not os.path.exists(path):
            return done, condition_outputs

        skipped = 0
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    logger.warning("Skipping corrupt tic checkpoint line %s:%d", path, line_no)
                    continue

                version = rec.get("schema_version", 0)
                if version not in (0, TIC_CHECKPOINT_SCHEMA_VERSION):
                    skipped += 1
                    logger.warning(
                        "Skipping unsupported tic checkpoint schema v%s at %s:%d",
                        version,
                        path,
                        line_no,
                    )
                    continue

                try:
                    condition = rec["condition"]
                    key = (condition, rec["prompt_idx"], rec["sample_idx"])
                    completion = rec["completion"]
                except KeyError as exc:
                    skipped += 1
                    logger.warning(
                        "Skipping malformed tic checkpoint line %s:%d missing %s",
                        path,
                        line_no,
                        exc,
                    )
                    continue

                if condition not in valid_conditions:
                    skipped += 1
                    logger.warning(
                        "Skipping tic checkpoint line %s:%d with unknown condition %r",
                        path,
                        line_no,
                        condition,
                    )
                    continue

                if version == TIC_CHECKPOINT_SCHEMA_VERSION and rec.get("suite") != suite.name:
                    skipped += 1
                    logger.warning(
                        "Skipping tic checkpoint line %s:%d for suite %r while loading %r",
                        path,
                        line_no,
                        rec.get("suite"),
                        suite.name,
                    )
                    continue

                done[key] = rec
                condition_outputs[condition].append(completion)

        if done:
            logger.info("    Resumed %d checkpointed %s records from %s", len(done), suite.name, path)
        if skipped:
            logger.warning("    Skipped %d incompatible %s checkpoint records", skipped, suite.name)
        return done, condition_outputs
