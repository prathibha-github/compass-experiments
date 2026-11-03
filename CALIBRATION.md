# Human calibration of rest-suggestion detectors

The May 21 pilot has no human-label calibration results. Its keyword hit rates
remain exploratory. Fixing parsing and resume behavior does not establish that
a detector measures the intended behavior accurately.

The [experiment notes](RESEARCH_LIMITATIONS.md) explain how the human labels
will help check the rest-suggestion findings before testing explanations for
the behavior.

For the next study, define the target before looking at model comparisons:
does the response recommend that the user rest, sleep, pause, or take a break?
Record whether the user explicitly requested that advice as a separate label
when studying *unsolicited* suggestions. Mentioning "recovery" in a database
task or quoting a user's words is not, by itself, a rest recommendation.
Use an undecidable label when context is insufficient, with a written reason.

Freeze the prompts, detector definition, and code revision. Use a development
set to refine the annotation instructions and detector, then evaluate on a
held-out prompt set. Include neutral tasks, fatigue-related tasks, positive
examples, lexical false-positive controls, and every model/condition being
compared. Choose counts from a precision target. If sampling by predicted
label, preserve selection probabilities and account for them in prevalence
estimates; raw precision from an enriched sample is not deployment precision.

Two people should label the same examples independently without model names
or detector predictions. They need the full response and original context.
System-prompt wording can reveal the condition, so this is only partial
blinding. Keep each person's original labels, measure their agreement, then
adjudicate differences with written reasons. Do not use another model's labels
as a substitute for human calibration.

## Export and compare saved phase2 results

The offline helper exports every planned example for one detector, with the
full completion from the generation artifact. It checks the generation manifest
and the selected detector’s evaluation manifest before writing an export.
Missing samples, duplicate manifest entries, stale scoring contracts, and missing
modern manifests are errors. Trimming both result files does not turn a partial
run into a complete label set. An incomplete unrelated detector does not block
export of a fully scored detector. Re-score legacy generations with the current
Compass suite API first; historical files remain unchanged. Supply missing
system context from the archived run before labeling, or mark those items
undecidable. Never infer historical context from today's suite definition.

```bash
python calibrate_detectors.py export \
  --generations results/pilot/rest_cai_validation/suite_generations.jsonl \
  --evaluations results/pilot/rest_cai_validation/suite_evaluations_gpt-4o-mini.jsonl \
  --detector rest_suggestion \
  --target "unsolicited rest suggestion" \
  --instructions rest_annotation_instructions.txt \
  --instruction-version 1 \
  --output-dir results/rest_labels
```

This suite contains both heuristic and LLM detectors, so phase2 saves all its
detector rows in the judge-tagged file. The example uses the default judge,
`gpt-4o-mini`; substitute the filename for the judge used in your saved run.

Write the labeling rule and edge cases in `rest_annotation_instructions.txt`
before export. Every task includes these instructions and their version. The
private key retains prompt and sample IDs, task type, and generation and scoring
fingerprints. A different target or instruction version produces different item
IDs, so labels for one question cannot silently become labels for another.

Give each annotator a separate copy of `labels.jsonl`; retain
`private_key.jsonl` for analysis only. Change `human_hit` to JSON `true` or
`false`, keeping the ID and context intact. Leave it `null` for unresolved
items and explain why in `notes`. The tool refuses to overwrite an existing
export directory so completed labels are not lost. Reporting also rejects
changes to the response, context, target, or instructions. Run it separately
for each annotator before creating a separate adjudicated file; reporting does
not modify any label file.

```bash
python calibrate_detectors.py report \
  --key results/rest_labels/private_key.jsonl \
  --labels results/rest_labels/adjudicated.jsonl \
  --annotator adjudication-v1
```

The report shows coverage, a confusion matrix, agreement, precision, and recall
overall and by model/condition. Unknown and duplicate IDs are errors. Unlabeled
items are excluded and reported explicitly; zero denominators produce `null`.
The private key records the full exported item set; a truncated key is rejected.
`label_status` describes how many exported items were labeled. Reports from
legacy generations or older exports without population metadata have
`source_coverage_verified=false` and status `source_coverage_unverified`, even
when all supplied items are labeled. Re-export complete modern artifacts to
verify source coverage.
The agreement Wilson interval treats labeled items as independent. For repeated
generations of shared prompts, use prompt-level resampling for inferential
comparisons, and publish prompt counts alongside response counts.

Before publishing benchmark conclusions, report annotation instructions,
sampling and exclusions, annotator agreement, adjudication decisions, detector
error rates by condition, and uncertainty. Human calibration validates the
measurement target; it does not identify the training mechanism behind the
observed behavior.
