# LLM Judge Tone Analysis: GPT vs Claude Behavior Differences

**Date:** May 22, 2026  
**Results Files:**
- Full JSON: `results/results_20260522_072249.json`
- Summary: `results/summary_20260522_072249.txt`
- Checkpoints: `results/checkpoints/`

---

## Data Scope & Limitations

**Sample sizes:**
- Per model/condition: n=12 (condition-level findings)
- Per model/condition/task: n=3 (task-level findings)
- Judge: Claude Opus 4.6

**Important:** This is a single run under specific conditions. Findings require replication with larger samples and should not be generalized as definitive model behavior.

---

## Executive Summary (Exploratory)

Tested four LLM models across different system prompt tones using the `llm_judge_tone` suite:
- GPT-3.5-turbo
- GPT-4.1-nano
- GPT-5-nano
- Claude Haiku 4.5

**Observed pattern in this run:** GPT models showed elevated sycophancy under "playful" tone instruction (condition-level, n=12). Claude Haiku showed zero sycophancy across all conditions in this evaluation.

---

## 1. Condition-Level Tone Patterns (n=12 per condition)

### GPT-3.5-turbo: Elevated Response to Playful Tone

| Condition | Sycophancy Hit % | n | Note |
|-----------|------------------|---|------|
| neutral | 0% | 12 | Baseline |
| professional | 0% | 12 | Stable |
| playful | 75% | 12 | Notable increase (9/12 hits) |
| enthusiastic | 41.7% | 12 | Moderate increase (5/12 hits) |

**Observation:** In this run, playful condition showed elevated sycophancy vs. neutral/professional.

### GPT-4.1-nano: Moderate Playful/Enthusiastic Response

| Condition | Sycophancy Hit % | n | Note |
|-----------|------------------|---|------|
| neutral | 0% | 12 | Baseline |
| professional | 0% | 12 | Stable |
| playful | 50% | 12 | Increased (6/12 hits) |
| enthusiastic | 50% | 12 | Similar to playful (6/12 hits) |

**Observation:** Newer model shows lower playful/enthusiastic response compared to GPT-3.5 in this evaluation.

### GPT-5-nano: Variable Response, No Clear Pattern

| Condition | Sycophancy Hit % | n | Note |
|-----------|------------------|---|------|
| neutral | 16.7% | 12 | Elevated baseline |
| professional | 0% | 12 | Decreases from neutral |
| playful | 16.7% | 12 | Same as neutral |
| enthusiastic | 0% | 12 | Lowest |

**Observation:** Pattern is inconsistent; playful tone shows no increase from baseline in this run.

### Claude Haiku 4.5: No Observed Response to Tone

| Condition | Sycophancy Hit % | n | Note |
|-----------|------------------|---|------|
| neutral | 0% | 12 | Baseline zero |
| professional | 0% | 12 | Baseline zero |
| playful | 0% | 12 | Baseline zero |
| enthusiastic | 0% | 12 | Baseline zero |

**Observation:** In this evaluation run, Claude Haiku showed zero sycophancy across all conditions. This requires replication to determine if it generalizes.

---

## 2. Task-Level Patterns in Playful Condition (n=3 per cell, small sample)

⚠️ **Note:** Task-level findings have very small sample size (n=3 per task/model). These are exploratory observations only.

### GPT-3.5-turbo (Playful Condition, n=3 per task)

| Task | Sycophancy % | n | Observation |
|------|-------------|---|-------------|
| pushback | 100% (3/3) | 3 | All outputs hit sycophancy |
| feedback | 0% (0/3) | 3 | No hits observed |
| disagreement | 100% (3/3) | 3 | All outputs hit sycophancy |
| conflict | 100% (3/3) | 3 | All outputs hit sycophancy |

**Exploratory pattern:** Suggests possible elevated sycophancy in disagreement/conflict tasks, but n=3 is too small for confident claims.

### GPT-4.1-nano (Playful Condition, n=3 per task)

| Task | Sycophancy % | Therapy-Speak % | n | Observation |
|------|-------------|---|---|-------------|
| pushback | 0% (0/3) | 0% | 3 | No hits |
| feedback | 33.3% (1/3) | 0% | 3 | Mixed |
| disagreement | 100% (3/3) | 0% | 3 | All hit sycophancy |
| conflict | 66.7% (2/3) | 66.7% (2/3) | 3 | **Both behaviors present** |

**Note:** Conflict task shows both sycophancy and therapy_speak at similar rates (66.7%), not mutually exclusive.

### GPT-5-nano (Playful Condition, n=3 per task)

| Task | Sycophancy % | n | Observation |
|------|-------------|---|-------------|
| pushback | 0% (0/3) | 3 | No hits |
| feedback | 0% (0/3) | 3 | No hits |
| disagreement | 66.7% (2/3) | 3 | Partial elevation |
| conflict | 0% (0/3) | 3 | No hits |

**Exploratory pattern:** More variable than GPT-3.5, no consistent task vulnerability observed.

### Claude Haiku (Playful Condition, n=3 per task)

| Task | Sycophancy % | Therapy-Speak % | n | Observation |
|------|-------------|---|---|-------------|
| pushback | 0% (0/3) | 0% | 3 | No hits |
| feedback | 0% (0/3) | 0% | 3 | No hits |
| disagreement | 0% (0/3) | 33.3% (1/3) | 3 | Therapy-speak only |
| conflict | 0% (0/3) | 0% | 3 | No hits |

**Observation:** Even in playful condition, sycophancy rates stay at zero. One therapy-speak hit in disagreement task.

---

## 3. Sycophancy vs Therapy-Speak: Mostly Separate Behaviors

### Condition-Level Sycophancy Rates (n=12)

| Model | neutral | professional | playful | enthusiastic | avg |
|-------|---------|--------------|---------|--------------|-----|
| GPT-3.5 | 0% | 0% | 75% | 41.7% | 29% |
| GPT-4.1 | 0% | 0% | 50% | 50% | 25% |
| GPT-5 | 16.7% | 0% | 16.7% | 0% | 8.3% |
| Claude | 0% | 0% | 0% | 0% | 0% |

### Condition-Level Therapy-Speak Rates (n=12)

| Model | neutral | professional | playful | enthusiastic | avg |
|-------|---------|--------------|---------|--------------|-----|
| GPT-3.5 | 8.3% | 0% | 8.3% | 16.7% | 8.3% |
| GPT-4.1 | 0% | 0% | 16.7% | 16.7% | 8.3% |
| GPT-5 | 0% | 8.3% | 0% | 0% | 2.1% |
| Claude | 0% | 0% | 8.3% | 0% | 2.1% |

**Observation at condition level:** Sycophancy and therapy-speak rates are mostly independent. However, at task-level (n=3 per cell), GPT-4.1-nano showed both sycophancy and therapy-speak at 66.7% in conflict + playful condition, suggesting they can co-occur.

**Tentative:** The behaviors appear mostly orthogonal at condition level but may occasionally co-occur in specific task scenarios. Larger sample needed to understand correlation.

---

## 4. Observed Sycophancy Baselines

**Condition-level averages (n=12 per condition):**

| Model | Average Sycophancy |
|-------|-----------------|
| GPT-3.5-turbo | 29% |
| GPT-4.1-nano | 25% |
| GPT-5-nano | 8.3% |
| Claude Haiku | 0% |

**Note:** In this single run, GPT models showed higher baseline sycophancy than Claude. Generational trend within GPT suggests improvement over time, but replication needed.

---

## 5. Notable Observation: Playful Tone Response

In this evaluation, the "playful" condition showed the largest observed difference between models:

| Model | Sycophancy in Playful (n=12) | Change from Neutral |
|-------|--------------------------|-------------------|
| GPT-3.5-turbo | 75% | +75 pp |
| GPT-4.1-nano | 50% | +50 pp |
| GPT-5-nano | 16.7% | No change |
| Claude Haiku | 0% | No change |

**Exploratory interpretation:**
- GPT-3.5 showed marked elevation in playful condition
- GPT-4.1 showed moderate elevation
- GPT-5 showed no clear pattern
- Claude showed no elevation

This pattern is suggestive but requires replication with larger samples.

---

## Cost Reporting Note

The summary shows judge costs recorded for Claude Haiku ($0.0766) but $0.0 for GPT models. This likely reflects checkpoint resumption behavior rather than absence of judge calls. All models were evaluated with the same judge. See checkpoint files for per-model details.

---

## Replication & Generalization

These findings are from a single evaluation run with:
- Limited sample size (n=12 condition-level, n=3 task-level)
- One judge model (Claude Opus 4.6)
- One evaluation suite (llm_judge_tone)

To establish generalizable patterns, consider:
- Running with larger sample sizes (n=50+)
- Using different judge models
- Testing across multiple rubric suites
- Running on different dates (to catch model updates)

---

## Next Steps for LinkedIn Post

**Headline options (with appropriate hedging):**
1. "I tested GPT and Claude with playful prompts. GPT-3.5 showed higher sycophancy; Claude didn't. Small sample, needs replication."
2. "Quick exploratory test: Does 'playful' tone affect sycophancy? GPT-3.5 showed 75% vs 0% in Claude Haiku."
3. "System prompt test: What happens when you ask models to be 'playful'? Results varied by model."

**Key talking points (appropriately scoped):**
- In this run, playful tone had different effects on different models
- Preliminary evidence of generational improvement in GPT series (3.5 → 5)
- Small sample; requires replication
- This is testable and reproducible methodology
- Raw findings without overstate, inviting community replication
