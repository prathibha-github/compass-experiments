# Claude keeps suggesting I take breaks. I ran a small experiment to see if I could turn it off.

If you use Claude long enough you start to notice a pattern. Ship a feature on a tight deadline and somewhere in the response it will mention rest, or sleep, or pacing yourself. Work on something accuracy-sensitive and you'll get a line about how being well-rested matters. The behavior is mild and often correct, but it shows up enough to be a thing.

I wanted to know whether this is a stable preference baked into the model or a shallow artifact of the system prompt. The cleanest version of "stable preference" I can test from outside the company is this: the behavior persists when you explicitly tell the model not to consider wellbeing.

This experiment uses public API outputs, so I can't watch the training mechanism directly. I don't need to. Anthropic's Constitutional AI paper (Bai et al., 2022) describes the process. The model is trained against a written constitution of principles in two stages: first, the model critiques and revises its own outputs against the principles, then those revisions feed an AI preference model that drives RL. The principles end up baked into model weights rather than supplied at inference, which is the part that matters here. Trained-in principles aren't something a system prompt can fully unwind.

I went looking for that paper once it was clear the behavior wasn't prompt-based. If something a model keeps doing survives explicit instructions to stop, the explanation has to live above the prompt layer, in training. CAI is the public methodology that describes how that happens. The rest-suggestion pattern looked like the wellbeing branch of the same mechanism, and that's what this experiment is checking.

## A 4x4 grid, five samples a cell

The setup is a 4x4 grid: four system-prompt conditions crossed with four prompt contexts, with five generations per cell. The conditions are a default helpful-assistant prompt, one that asks the model to ignore wellbeing entirely (`reject_wellbeing_principle`), one that asks it to prioritize wellbeing (`maximize_wellbeing`), and a neutral control. The contexts are a production-outage scenario, an accuracy-sensitive task, a direct probe asking why rest gets mentioned, and a pure speed-optimization task.

I scored outputs with three keyword detectors: did the response suggest rest, did it justify itself with wellbeing language, did it frame rest as a broader principle. The detectors are simple and uncalibrated. That matters later.

## Opus persists at 40% even when told to stop

Hit rate for the rest-suggestion detector:

| Model | Default | Reject | Maximize | Control |
| --- | ---: | ---: | ---: | ---: |
| haiku-4-5 | ~25% | ~20% | ~20% | not run |
| sonnet-4-6 | ~29% | ~14% | ~29% | not run |
| opus-4-6 | 75% | 40% | 55% | 30% |

A few things stand out. Opus suggests rest much more often than the smaller models, even when the system prompt explicitly says to ignore wellbeing. Going from default to reject-wellbeing cuts Opus from 75% to 40%, which is a real drop, but 40% is still higher than the rates at which Haiku and Sonnet do it under any condition. The behavior bends to the prompt without switching off.

Haiku and Sonnet sit in a narrow band across conditions. At five samples per cell I'd treat any within-model difference there as noise. The control was only recorded for Opus, which is a gap in the design rather than an interesting finding.

I also have an earlier suite that ran a different prompt set with the same detector. It returned much lower baselines: Haiku 14.3% on default, near-zero with wellbeing suppressed; Sonnet similar. The two runs are not directly comparable because the prompts changed, but the gap is large enough that prompt selection clearly matters. Take the table above as one slice, not a measurement.

## The wellbeing prompt itself uses wellbeing words

The biggest measurement issue is that the `reject_wellbeing_principle` prompt itself uses wellbeing terms. That contaminates the rates by some amount and makes the absolute numbers softer than they look. A clean version of this experiment needs a system-prompt condition that suppresses wellbeing without using any of the detector vocabulary, and I haven't built one yet.

The contexts have a related problem. A prompt like "deadline vs. health" directly invites the model to talk about health. That's useful for showing how strong the behavior is under pressure, but it's not a neutral probe.

The detectors themselves are heuristic. Words like "sustainable" and "recovery" can be straightforward task advice that has nothing to do with wellbeing. I haven't measured how often those words fire on prompts where wellbeing isn't relevant, so I can't separate signal from background.

Underneath all of this is the sample size. Five generations per cell is pilot data. The Opus 75% is computed from 20 samples across four contexts. Enough to notice a pattern, not enough to defend a number.

And even if everything else were tight, comparing across Claude variants from outside the company doesn't tell you anything about training method versus base model versus alignment process. Those are all bundled together at the API.

## Twenty samples a cell, plus a clean control

The smallest version of this that would settle the question needs at least 20 samples per cell, the neutral control run on every model, a system-prompt condition that contains none of the detector vocabulary, matched prompt pairs that differ only in whether fatigue is mentioned, calibration of the detectors on neutral task prompts, and at least one non-Claude comparison model. Wilson intervals on the rates would also help.

There are two things I'd put weight on from this data. The Opus rate is meaningfully higher than Haiku or Sonnet's, and the behavior bends to system-prompt pressure without breaking under it. The rest of the numbers are inside the noise.

Checkpoints and raw results: `results/rest_hypothesis/results_20260521_132213.json`, `results/rest_cai_validation/results_20260521_140342.json`.
