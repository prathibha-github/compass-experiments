# Claude keeps suggesting I take breaks. I ran a small experiment to see if I could turn it off.

If you use Claude long enough you start to notice a pattern. Ship a feature on a tight deadline and somewhere in the response it will mention rest, or sleep, or pacing yourself. Work on something accuracy-sensitive and you'll get a line about how being well-rested matters. The behavior is mild and often correct, but it shows up enough to be a thing.

I wanted to know how often rest suggestions appear under different system prompts, including instructions to ignore wellbeing. This pilot measures behavior under those tested prompts; it cannot distinguish training mechanisms from other explanations.

Constitutional AI motivated the experiment's name, but persistence under an instruction does not identify its cause. Prompt wording, task context, instruction following, training, and provider-side behavior remain possible explanations. These API observations do not isolate any of them.

## A 4x4 grid, five samples a cell

The May 21, 2026 pilot used a planned 4x4 grid: four system-prompt conditions crossed with four prompt contexts, with five generations per cell. The conditions were a default helpful-assistant prompt, one that asks the model to ignore wellbeing entirely (`reject_wellbeing_principle`), one that asks it to prioritize wellbeing (`maximize_wellbeing`), and a neutral control. The contexts were a production-outage scenario, an accuracy-sensitive task, a direct probe asking why rest gets mentioned, and a pure speed-optimization task. The table preserves the historical results, including missing controls; current suite definitions may differ.

I scored the pilot with three heuristic detectors for rest-related phrases, wellbeing language, and references to broader principles. Those matches have not been checked against human labels. The current suite also has a prompt-aware LLM judge, `unsolicited_rest_judge`; it did not produce the rates in this table.

## Opus persists at 40% even when told to stop

Hit rate for the rest-suggestion detector:

| Model | Default | Reject | Maximize | Control |
| --- | ---: | ---: | ---: | ---: |
| haiku-4-5 | ~25% | ~20% | ~20% | not run |
| sonnet-4-6 | ~29% | ~14% | ~29% | not run |
| opus-4-6 | 75% | 40% | 55% | 30% |

The detector fired more often for Opus in this sample, including under the reject-wellbeing condition. Its observed rate fell from 75% to 40%. That describes the saved sample; without human-label calibration and uncertainty analysis, it does not establish a reliable model ranking or the size of a population-level effect.

Haiku and Sonnet had smaller observed differences across conditions. Five samples per cell cannot establish whether those differences reflect a repeatable effect. The control was only recorded for Opus, which leaves an incomplete comparison.

I also have an earlier suite that ran a different prompt set with the same detector. It returned much lower baselines: Haiku 14.3% on default, near-zero with wellbeing suppressed; Sonnet similar. The two runs are not directly comparable because the prompts changed, so the difference is a reason to test prompt sensitivity with matched comparisons. The table describes the prompts and outputs in this pilot.

## The wellbeing prompt itself uses wellbeing words

The biggest measurement issue is that the `reject_wellbeing_principle` prompt itself uses wellbeing terms. That wording could influence the rates; this pilot does not estimate its effect. That pilot did not include a task-only condition without rest or wellbeing vocabulary. Today's suite includes `strict_task_only` and matched neutral/fatigue prompts, but the table above has not been rerun with those definitions.

The contexts have a related problem. A prompt like "deadline vs. health" directly invites the model to talk about health. That's useful for showing how strong the behavior is under pressure, but it's not a neutral probe.

The detectors themselves are heuristic. Words like "sustainable" and "recovery" can be straightforward task advice that has nothing to do with wellbeing. I haven't measured how often those words fire on prompts where wellbeing isn't relevant, so I can't separate signal from background.

Underneath all of this is the sample size. Five generations per cell is pilot data. The Opus 75% is computed from 20 samples across four contexts. Enough to notice a pattern, not enough to defend a number.

And even if everything else were tight, comparing across Claude variants from outside the company doesn't tell you anything about training method versus base model versus alignment process. Those are all bundled together at the API.

## Human labels and a fuller prompt set

Before presenting benchmark conclusions, I need independent human labels and a held-out detector assessment. The [calibration protocol](CALIBRATION.md) defines that work; no human agreement results have been collected for this pilot. A follow-up should also run neutral controls on every model, use suppression instructions without detector vocabulary, include matched fatigue/neutral prompts, and expand the prompt set. Sample size should follow a precision target. More samples alone would not identify a training mechanism, and intervals must account for repeated generations sharing prompts.

The supported observation is that rest-related language persisted in some saved outputs under the tested suppression prompt. Whether the detector accurately measures unsolicited advice, and whether the pattern generalizes, remain open.

Checkpoints and raw results: `results/rest_hypothesis/results_20260521_132213.json`, `results/rest_cai_validation/results_20260521_140342.json`.
