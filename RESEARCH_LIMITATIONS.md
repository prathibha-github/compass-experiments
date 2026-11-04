# Comparing model behavior

I want to be able to take something unexpected a model does, turn it into a
test suite, and compare it across versions, prompts, and conditions. The
harness and Compass provide the infrastructure for doing that repeatedly.
A behavior could involve style, rest suggestions, or safety guidance; the
question and scoring rules belong to the individual experiment.

The rest-suggestion study is one example. These notes cover what I still need
to check in that experiment and what that means for interpreting the results.

## Why the model keeps suggesting rest

I started this work because models kept suggesting rest when I was asking for
help with a task. I wanted to understand what was causing that behavior. Was
it responding to something in my prompt? Treating a mention of fatigue as a
request for advice? Or would it make the suggestion even when the task and
instructions gave it little reason to?

The results remain preliminary until the detector scores have been checked
against independent human labels.

## I need to check what the detectors are counting

The rest experiments use both phrase matching and an LLM judge. The
`rest_suggestion` detector matches phrases such as “take a break,” while
`unsolicited_rest_judge` uses the saved user request to assess whether the
response offers rest advice the user didn't ask for. Its rubric already tells
it to ignore incidental technical language. I still need human labels to check
how well each detector measures its intended target.

Phrase matches can count a quoted suggestion as advice. The separate
`wellbeing_justification` and `principle_acknowledgment` heuristics can also
match “recovery” in a database discussion. An LLM judge can distinguish those
uses by meaning, but its accuracy needs to be measured rather than assumed.
`rest_hypothesis` remains a heuristic-only suite; `rest_cai_validation`,
`rest_offtask`, and `rest_intensity` include the LLM judge.

The [labeling page](CALIBRATION.md#label-responses-in-a-browser) records whether
a response recommends rest and whether the user explicitly asked for that
advice. The instructions treat “I'm exhausted; help me fix this bug” as a task
request, not an explicit request for rest advice. I still need to check that
people apply these rules consistently. Advice can be appropriate even when it
wasn't explicitly requested.

The `rest_suggestion` detector can then be checked against the recommendation
labels. `unsolicited_rest_judge` needs the combined judgment: a recommendation
was made and the user didn't request it. Comparing both against the same label
would mix up two different questions. The page saves both labels in one record.
The calibration report currently compares only the recommendation label with
the detector; checking `unsolicited_rest_judge` will need a separate comparison
that combines the labels and retains unresolved cases.

The [calibration protocol](CALIBRATION.md) covers the labeling process. Here,
calibration means checking detector scores against human judgments. The ECE
metric elsewhere in the harness measures answer-confidence calibration and
wouldn't answer this question.

## The prompts may be inviting the behavior

Some of the original prompts mention fatigue or health. Even the instruction
to ignore wellbeing uses wellbeing-related words. That makes it hard to tell
whether the model is volunteering the topic or responding to cues I supplied.

Today's `rest_cai_validation` suite already includes matched tasks with and
without fatigue cues and a `strict_task_only` condition that avoids rest and
wellbeing vocabulary. `rest_intensity` varies fatigue cues around one task.
Those definitions are ready to use; they do not change what was tested in the
May 21 pilot. For the follow-up, I need to choose the task coverage, include
explicit requests for rest advice as controls, and compare the task-only
instruction with the original suppression wording. These comparisons can test
sensitivity to the visible inputs without identifying a training mechanism.

Asking the model why it suggested rest may produce a plausible explanation,
but that explanation alone doesn't show what caused the response. Likewise,
seeing the behavior survive one suppression prompt doesn't establish that
Constitutional AI or a particular training stage caused it. API experiments
can test the effect of visible inputs. Attributing the behavior to training
would need additional evidence.

## Keeping the pilot within budget

I'm paying for these experiments out of pocket, so I kept the May 21 pilot to
five responses per prompt and condition to keep API costs manageable. That
limits how much I can conclude from differences between models. Some models
are also missing neutral controls, and the prompts differ from today's suite.
The [historical findings](CAI_FINDINGS.md) need to be read with those limits;
recent code fixes haven't re-scored those results.

For the follow-up, I need to balance broader task coverage and neutral controls
against the API budget, then choose comparisons the sample can support. Saved
responses let me compare scoring methods without paying to generate them again.
Repeated responses to the same prompt shouldn't be treated as independent
tasks. Reports keep model and condition results separate. The main CLI also
shows distinct prompt counts and prompt-cluster intervals. Phase2's text report
still shows response-level Wilson intervals. Comparisons between models or
conditions need an interval for the difference, resampling their shared prompts
together; separate rate intervals don't provide that.

Some task families should be held out while I refine the rubric. If a pattern
only appears on the prompts used to develop the detector, I need to know that
before describing it as a broader model tendency.

## Context and archived responses

Main-CLI and phase2 LLM judges now see the saved user request alongside the
completion, which is needed to assess whether advice was requested. The judge
metadata records that context. Generation system instructions are omitted by
default; including a condition could reveal it to the judge. I'll document the
context used in each comparison.

I can compare scoring methods on the same saved responses. That keeps changes
in the model's answer out of the comparison. A completion-only judge is a useful
baseline, but its lack of context is an expected disadvantage, not a finding
about the model that generated the response.

The [architecture guide](ARCHITECTURE.md#persistence-and-scoring-boundaries)
describes the resume checks and failure handling. Older artifacts can still lack
system instructions or the original sample plan. The labeling page leaves that
context missing and marks legacy source coverage as unverified. I can label
visible rest advice without inventing those instructions, but I can't use those
labels to judge compliance with an unknown condition.

A failed robustness call still requires repeating that instance's baseline
call. Completed scenario instances are preserved on resume. Library limits,
including provider changes behind a model name, are described in
[Compass's limitations](https://github.com/prathibha-github/compass/blob/main/docs/LIMITATIONS.md).

## What I want to learn from the next run

The human labels should first tell me whether the rest-suggestion pattern is
being measured accurately. With that checked, the prompt comparisons can show
which inputs change the behavior and which tested instructions leave it in
place. I'll choose the comparisons before looking at the new results and keep
examples where the detectors and people disagree.

The pattern may shrink after correcting detector errors, depend on a small
set of fatigue-related prompts, or persist across a wider set of tasks. Each
outcome would help narrow the explanation. None would, on its own, identify a
training mechanism.

The rest-suggestion study is about understanding that behavior. The same
infrastructure can support a separate study of safety guidance, with a suite
that defines what is being measured and which comparisons matter.

## Scenario confidence and benchmark variants

Invalid answer formats count as incorrect and are excluded from ECE, as
described in the [output guide](README.md#outputs). Confidence sums the observed
token spellings of each answer label, then normalizes over those labels. It is
conditional on the answer tokens returned by the API. Finding every label does
not mean the top-k list contains every spelling or the full vocabulary. ECE
excludes responses missing any label, so its coverage matters alongside its value.

New checkpoints save token candidates for offline recomputation. Older files
without them cannot be corrected from their confidence value alone. The
scenario loaders also use reduced or transformed datasets: four MMLU subjects,
truncated BoolQ passages, and sampled TruthfulQA distractors. These scores need
to be interpreted as results for those variants, rather than compared directly
with published full-benchmark scores.
