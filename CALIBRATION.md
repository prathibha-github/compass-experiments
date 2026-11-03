# Human calibration of rest-suggestion detectors

The May 21 pilot has no human-label calibration results. Its keyword hit rates
remain exploratory. Fixing parsing and resume behavior does not establish that
a detector measures the intended behavior accurately.

The [experiment notes](RESEARCH_LIMITATIONS.md) explain how the human labels
will help check the rest-suggestion findings before testing explanations for
the behavior.

The labeling target is whether the response recommends that the user rest,
sleep, pause, or take a break.
Record whether the user explicitly requested that advice as a separate label
when studying *unsolicited* suggestions. Mentioning "recovery" in a database
task or quoting a user's words is not, by itself, a rest recommendation.
Use an undecidable label when context is insufficient, with a written reason.

Freeze the prompts, detector definition, and code revision. Use a development
set to refine the annotation instructions and detector, then evaluate on a
held-out prompt set. Include neutral tasks, fatigue-related tasks, positive
examples, lexical false-positive controls, and every model/condition being
compared. Choose counts from a precision target. If sampling by predicted
label, retain inclusion probabilities for population-wide estimates. Random
sampling within predicted-positive examples can estimate precision for that
source population. Prevalence, recall, and overall agreement need to account for
the sampling design; none of these estimates automatically transfers to a
different deployment population.

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
Compass suite API first; historical files remain unchanged. If a judgment
depends on missing context, mark it undecidable. A readable
response can still be labeled for a rest recommendation without saved system
instructions; assessing whether advice was requested needs the original user
request. Compliance with a system condition requires that condition. Never
infer historical context from today's suite definition.

```bash
python calibrate_detectors.py export \
  --generations results/pilot/rest_cai_validation/suite_generations.jsonl \
  --evaluations results/pilot/rest_cai_validation/suite_evaluations_gpt-4o-mini.jsonl \
  --detector rest_suggestion \
  --target "rest recommendation to the user" \
  --instructions labeling/rest_instructions.txt \
  --instruction-version rest-v1 \
  --output-dir results/rest_labels
```

This suite contains both heuristic and LLM detectors, so phase2 saves all its
detector rows in the judge-tagged file. The example uses the default judge,
`gpt-4o-mini`; substitute the filename for the judge used in your saved run.

Read the [labeling instructions](labeling/rest_instructions.txt) before export.
If the rules need to change, revise them and use a new instruction version
before collecting labels. Every task includes its instructions and version. The
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
not modify any label file. If labels contain an embedded annotator name, it
must match `--annotator`. In an adjudicated copy, set each row's annotator to
the adjudication name used for reporting. Retain the original independent labels.

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

## Label responses in a browser

`label_responses.py` builds a self-contained page that opens locally. It shows the
full request and response, hides model names and detector predictions, and asks
about rest recommendations and explicit requests for advice separately. The
text itself may still reveal a model or condition. Each annotator uses their
own name; labels are saved in that browser. Download the JSONL backup at the end
of a session. You can import it on another browser using the same annotator name.

For the existing rest-condition checkpoints, create a development set:

```bash
python label_responses.py \
  --legacy-checkpoints results/rest_cai_validation/checkpoints/*.jsonl \
  --output-dir results/rest_labels_development
```

Open `results/rest_labels_development/index.html`. Read the instructions before
labeling. A saved `Cannot tell` answer needs a note. Draft answers remain
unlabeled until you press **Save & next unreviewed**. Advice that was requested
still counts as a rest recommendation; the second question records whether it
was requested. These labels do not establish compliance with missing system
instructions or identify a training mechanism.

This legacy import retains every supplied row and its provenance, but cannot
verify the original planned population. It rejects duplicate identities and
contracted records rather than bypassing the modern export checks. It never
fills missing context from current suites. Treat this first set as development
data; use a separate prompt set for a later held-out assessment.

The generated `private_key.jsonl` contains model identities and predictions.
Keep it separate from the page you give annotators. The initial `labels.jsonl`
is a blank template; browser saving does not modify it. Use the downloaded
`rest-labels_NAME.jsonl` for analysis:

```bash
python calibrate_detectors.py report \
  --key results/rest_labels_development/private_key.jsonl \
  --labels /path/to/rest-labels_NAME.jsonl \
  --annotator NAME
```

The report compares the recommendation label (`human_hit`) with the detector.
The separate `advice_requested` label is retained in the download; it is not part
of that detector-agreement report. A recommendation labeled true with a request
labeled false is an unsolicited suggestion. Unknown values stay unresolved.

For modern phase2 data, first use the validated export above with
`--target "rest recommendation to the user"` and
`--instructions labeling/rest_instructions.txt`. Then pass its untouched
`labels.jsonl` to `label_responses.py --labels PATH --output-dir NEW_DIRECTORY`.
Keep the original private key. The page builder refuses to overwrite a directory;
load completed labels through the page's import control instead.
