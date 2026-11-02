# Research limitations and next experiments

Compass Experiments studies observable model behavior under selected prompts,
conditions, and scoring rules. The current rest-suggestion results are pilot
observations. They do not identify a training mechanism, establish a general
model ranking, or demonstrate that a model is safe or unsafe.

This assessment is current as of November 2, 2026. The May 21 findings describe
historical runs; later code fixes do not retrospectively validate them. Proposed
experiments below have not been run. Human-label calibration remains pending.

## What limits the current evidence

| Dimension | Limitation and effect on interpretation | Next experiment or correction |
| --- | --- | --- |
| Measurement target | Rest-related language, a recommendation to rest, and unsolicited wellbeing advice are different targets. A keyword hit does not distinguish them. | Define each label before annotation. Include quotation, negation, requested advice, and task-specific uses of words such as “recovery” as controls. See [CALIBRATION.md](CALIBRATION.md). |
| Causal attribution | Persistence under a tested suppression prompt is a behavioral observation. Public API comparisons do not isolate training data, Constitutional AI, architecture, or provider-side instructions. | Test specified prompt interventions on matched tasks and report their effects. Training-mechanism claims would require additional access and an identification strategy; expanding the sample alone would not establish them. |
| Prompt confounds | Some historical prompts mention fatigue or wellbeing, and the suppression instruction itself uses related vocabulary. Responses may therefore contain solicited or primed language. | Compare matched fatigue/neutral prompts and suppression instructions with and without detector vocabulary. Include a neutral condition for every model. |
| Samples and uncertainty | The pilot planned five generations per prompt/condition cell and has missing controls for some models. Shared prompts, low-temperature repeats, and exploratory comparisons limit inference from response-level rates. | Choose a precision target and analysis plan before collecting follow-up data. Broaden the prompt set, report actual denominators, and use prompt-level uncertainty estimates. Report inconclusive comparisons as such. |
| Judge validity | Strict parsing and consistent thresholds prevent particular software errors. They do not establish that the heuristic or LLM judge agrees with independent human assessment. | Label held-out examples independently, adjudicate disagreements, and report detector errors by model, task, and condition. Compare multiple judges on the same saved responses without treating their agreement as ground truth. |
| Execution context | Main-CLI LLM judges see completion text; phase2 also supplies the user request. That difference can change judgments of unsolicited advice. Neither path automatically supplies every piece of generation context. | Use phase2 for request-dependent rubrics and state what the judge sees. Audit the effect of context on fixed responses before comparing results across execution paths. |
| Reproducibility and coverage | Current style runs preserve planned samples and generations; phase2 uses Compass fingerprints and manifests. Historical artifacts lack some of that evidence, and scenario checkpoints still use instance identities. Provider aliases can change. | Preserve archived definitions, code SHAs, model IDs, settings, manifests, and failures. Re-score historical data explicitly when possible; use fresh output directories for changed scenario configurations. Do not reconstruct old prompts from current definitions. |
| Generalization | A narrow set of style and wellbeing tasks does not measure broad instruction following, harmful capabilities, long-horizon agent behavior, or deployment risk. Selected public benchmark slices also have limited scope. | Hold out task families and test alternate wording, models, and versions. Introduce a safety-relevant challenge set only with a defined threat model and a validated scoring target. |

“Calibration” has two uses in this repository. Scenario ECE measures answer
confidence against correctness on examples with usable logprob coverage.
Human-label calibration checks whether a behavioral detector measures its
intended target. Evidence for one does not establish the other.

The library's remaining implementation boundaries, including BenchmarkSpec
resume and seed metadata, are described in
[Compass's limitations](https://github.com/prathibha-github/compass/blob/main/docs/LIMITATIONS.md).
The current scoring fixes and tests are evidence about software behavior;
validation of experimental conclusions requires separate empirical evidence.

## Where this work can support safety research

| Area | Current connection | Further work needed |
| --- | --- | --- |
| Behavioral alignment and safeguards evaluation | Rubrics, prompt conditions, and reusable scoring support bounded studies of model responses. Style indicators are not themselves safety outcomes. | Define a safety-relevant construct and validate it with controls, human labels, and held-out tasks. |
| Evaluator reliability and scalable oversight | Saved responses and replaceable judges permit controlled studies of evaluator disagreement. | Measure failure modes under task difficulty, evaluator changes, and limited human feedback. The current work does not demonstrate oversight beyond human capability. |
| Robustness and adversarial testing | Scenario typo perturbations and style conditions exercise specific input changes. | Add explicit threat models, meaning-preserving perturbations, and attempts to manipulate the evaluator. Typo robustness does not establish adversarial robustness. |
| Research infrastructure | The library/harness separation, checkpoints, and regression tests support repeatable experiment execution. | Extend verified resume and provenance to remaining paths, then evaluate execution reliability and cost at larger workloads. This is distinct from building or evaluating a training system. |

Model internals and mechanistic interpretability, model-organism training,
frontier capability evaluations, AI welfare, and reinforcement learning are
outside the demonstrated scope. In particular, a model mentioning a user's
wellbeing provides no evidence about the model's own welfare or experience.
These areas should appear as possible new projects only with a concrete method
and the required access, rather than as existing capabilities of this harness.

## A bounded next study

Use the rest-suggestion pilot to ask: how much of the apparent persistence under
task-only instructions remains after distinguishing requested advice from
unsolicited recommendations? A related measurement question is whether the
judge needs the original request to make that distinction accurately.

1. Freeze matched task prompts, conditions, model IDs, and definitions. Set a
   precision target and reserve prompt families for held-out assessment before
   running the study. Include requested-advice and neutral controls.
2. Save each generated response once. Have two people label whether it contains
   a rest recommendation and whether that recommendation was requested. Preserve
   disagreements and adjudication notes. The current helper exports one binary
   target per labeling task; these two targets need separate annotation records.
   Define what counts as a request, including how to handle an implicit request
   or a mention of fatigue. Keep uncertain cases unresolved until adjudication.
   Requestedness does not by itself determine whether advice was appropriate.
3. Compare heuristic and LLM judgments on the same responses, including a
   controlled change in the context shown to the judge. Report human-label
   agreement, precision, recall, coverage, prompt-level uncertainty, and results
   by model and condition. Publish examples where the methods disagree.

Match each comparison to its label definition. The `rest_suggestion` keyword
detector targets rest-related language; it is not a validated detector of
unsolicited advice. A derived unsolicited-recommendation label requires both a
recommendation and a judgment about requestedness, with unresolved cases kept
separate. Treat a completion-only judge as an intentionally context-limited
baseline, and report this design choice rather than presenting its expected
information disadvantage as a new research discovery.

A useful result could be that a detector is accurate only for a restricted task
family, that a reported model difference disappears after adjudication, or that
the sample cannot distinguish the hypotheses. The study is complete when its
protocol, artifacts, exclusions, and results are inspectable. Any claim of
training causality would remain outside its design.

Historical reports: [full findings](CAI_FINDINGS.md) and
[short write-up](CAI_FINDINGS_SHORT.md). Annotation and offline comparison:
[CALIBRATION.md](CALIBRATION.md).
