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

The results are preliminary. I haven't checked the detector scores against
independent human labels yet.

## I need to check what the detectors are counting

Mentioning rest, recommending it, and giving advice the user didn't request
are different things. The current keyword detector can't reliably distinguish
them. “Recovery” could refer to a database, and a response can discuss a rest
suggestion without making one.

I plan to have two people independently label whether a response recommends
rest and whether the user asked for that advice. An example such as “I'm
exhausted; help me fix this bug” needs an agreed interpretation, or an uncertain
label. Advice can be appropriate even when it wasn't explicitly requested.

The `rest_suggestion` detector can then be checked against the recommendation
labels. `unsolicited_rest_judge` needs the combined judgment: a recommendation
was made and the user didn't request it. Comparing both against the same label
would mix up two different questions. The current labeling helper handles one
binary target at a time, so these need separate annotation records.

The [calibration protocol](CALIBRATION.md) covers the labeling process. Here,
calibration means checking detector scores against human judgments. The ECE
metric elsewhere in the harness measures answer-confidence calibration and
wouldn't answer this question.

## The prompts may be inviting the behavior

Some of the original prompts mention fatigue or health. Even the instruction
to ignore wellbeing uses wellbeing-related words. That makes it hard to tell
whether the model is volunteering the topic or responding to cues I supplied.

The next run should compare matched tasks with and without fatigue cues. I
also want to compare task-only instructions that avoid the detector vocabulary
with the original suppression wording, and include prompts that explicitly
ask for rest advice. Those comparisons can help separate sensitivity to the
wording from persistence across the tested instructions.

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
tasks. I'll keep model and condition results separate and account for shared
prompts when estimating uncertainty.

Some task families should be held out while I refine the rubric. If a pattern
only appears on the prompts used to develop the detector, I need to know that
before describing it as a broader model tendency.

## The scoring path can change the result

Main-CLI and phase2 LLM judges now see the saved user request alongside the
completion, which is needed to assess whether advice was requested. The judge
metadata records that context. Generation system instructions are omitted by
default; including a condition could reveal it to the judge. I'll document the
context used in each comparison.

I can compare scoring methods on the same saved responses. That keeps changes
in the model's answer out of the comparison. A completion-only judge is a useful
baseline, but its lack of context is an expected disadvantage, not a finding
about the model that generated the response.

Current style runs preserve planned samples and generated responses through
failures. Phase2 checks fingerprints and manifests before reusing work. Older
artifacts are missing some of that information. Scenario checkpoints now verify
formatted prompts, answer choices, reference answers, few-shot examples as
rendered in the prompt, and robustness settings before reusing results. They
also retain the planned instance set and report incomplete runs as failures.
Changed or legacy checkpoints need a fresh output directory. A completed
scenario instance is preserved on resume, but a failed robustness call still
requires repeating that instance's baseline call. I'll keep the archived prompts, settings, and code
revisions with each run rather than reconstruct old runs from current suite
definitions. Remaining library issues are in
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
