import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tqdm import tqdm

from helm_eval_lex_quirks.config import DEFAULT_SCENARIO_CONFIGS, EvalConfig, model_config
from helm_eval_lex_quirks.metrics.accuracy import extract_mc_answer, multiple_choice_accuracy
from helm_eval_lex_quirks.metrics.calibration import compute_ece, extract_choice_probs
from helm_eval_lex_quirks.metrics.quirk_detection import QUIRK_PROMPTS, SYSTEM_PROMPTS, find_quirks
from helm_eval_lex_quirks.metrics.linkedin_style import (
    LINKEDIN_SUITE,
    analyze_outputs,
    compute_uplift,
)
from helm_eval_lex_quirks.metrics.robustness import compute_robustness_score, inject_typos
from helm_eval_lex_quirks.metrics.tic_suite import TicSuite, analyze_text, summarize_outputs
from helm_eval_lex_quirks.models import CompletionClient, create_client
from helm_eval_lex_quirks.scenarios.base import Instance, Scenario
from helm_eval_lex_quirks.scenarios.bbq import BBQScenario
from helm_eval_lex_quirks.scenarios.boolq import BoolQScenario
from helm_eval_lex_quirks.scenarios.hellaswag import HellaSwagScenario
from helm_eval_lex_quirks.scenarios.mmlu import MMLUScenario
from helm_eval_lex_quirks.scenarios.truthfulqa import TruthfulQAScenario

logger = logging.getLogger(__name__)

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
            if self.config.run_linkedin_style:
                n_linkedin = (
                    len(LINKEDIN_SUITE.prompts)
                    * len(LINKEDIN_SUITE.conditions)
                    * self.config.linkedin_samples
                )
                total += estimate_model_cost(
                    model_name,
                    ["x" * 100] * n_linkedin,
                    max_tokens=LINKEDIN_SUITE.max_tokens,
                )
            estimates[model_name] = round(total, 4)
        return estimates

    def run(self) -> Dict[str, EvalResults]:
        out = {}
        for model_name in self.config.models:
            logger.info("Evaluating: %s", model_name)
            client = create_client(self.config, model_name)
            out[model_name] = self._run_model(client, model_name)
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

        if not self.config.linkedin_only:
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

        if self.config.run_linkedin_style:
            logger.info("  LinkedIn style detection...")
            result.linkedin_results = self._run_linkedin_style(client, model_name)

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

    def _run_linkedin_style(self, client: CompletionClient, model_name: str) -> dict:
        return self._run_tic_suite(
            client=client,
            model_name=model_name,
            suite=LINKEDIN_SUITE,
            samples=self.config.linkedin_samples,
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
        analyze_fn=None,
        uplift_fn=None,
    ) -> dict:
        condition_outputs: Dict[str, List[str]] = {c.name: [] for c in suite.conditions}
        safe_model = model_name.replace("/", "_").replace(":", "_")
        ckpt_path = os.path.join(
            self.config.output_dir,
            "checkpoints",
            f"{safe_model}__{suite.checkpoint_stem}.jsonl",
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        # Load checkpoint and track which (condition, prompt_idx, sample_idx) are done
        done: Dict[tuple, dict] = {}
        if os.path.exists(ckpt_path):
            with open(ckpt_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        key = (rec["condition"], rec["prompt_idx"], rec["sample_idx"])
                        done[key] = rec
                        if rec["condition"] in condition_outputs:
                            condition_outputs[rec["condition"]].append(rec["completion"])
                    except (json.JSONDecodeError, KeyError):
                        pass
            if done:
                logger.info(
                    "    Resumed %d checkpointed %s records from %s",
                    len(done),
                    suite.name,
                    ckpt_path,
                )

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
                            detector_results = analyze_text(completion, suite)

                            record = {
                                "model": model_name,
                                "suite": suite.name,
                                "condition": condition.name,
                                "prompt_id": prompt.id,
                                "prompt_idx": prompt_idx,
                                "sample_idx": sample_idx,
                                "prompt": prompt.text,
                                "completion": completion,
                                "detectors": {
                                    name: result.to_dict()
                                    for name, result in detector_results.items()
                                },
                            }
                            record.update(suite.extra_record_fields(completion, detector_results))
                            ckpt_f.write(json.dumps(record) + "\n")
                            ckpt_f.flush()
                        except Exception as exc:
                            logger.error(
                                "  %s prompt failed (%s/%s): %s",
                                suite.name,
                                condition.name,
                                prompt.text[:30],
                                exc,
                            )
        pbar.close()

        analysis = (
            analyze_fn(condition_outputs) if analyze_fn else summarize_outputs(condition_outputs, suite)
        )
        uplift = (
            uplift_fn(analysis, baseline=suite.baseline_condition)
            if uplift_fn
            else {}
        )
        return {
            "suite": suite.name,
            "analysis": analysis,
            "uplift": uplift,
            "checkpoint_path": ckpt_path,
        }
