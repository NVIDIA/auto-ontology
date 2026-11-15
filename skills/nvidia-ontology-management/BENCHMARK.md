<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Skill Benchmark: nvidia-ontology-management

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-ontology-management`
- Validation status: `passed`
- Evaluation date: 2026-11-15
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 13 evaluation tasks (13 positive)
- Dataset digest: `sha256:4c8455d88fc45fd16ded90344c740ea7149fe29edbca7f112b346c21177d23fa` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvidia-ontology-management | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-ontology-management | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `nvidia-ontology-management`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 87.8% — baseline ran, but no comparable score was available; uplift unavailable | 84.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 88.5% (-11.5 points) | 100.0% → 92.3% (-7.7 points) |
| Correctness | 38.5% → 95.4% (+56.9 points) | 49.2% → 81.5% (+32.3 points) |
| Discoverability | 95.4% — baseline ran, but no comparable score was available; uplift unavailable | 93.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 27.8% → 73.1% (+45.3 points) | 44.5% → 65.4% (+20.9 points) |
| Efficiency | 86.8% — baseline ran, but no comparable score was available; uplift unavailable | 88.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 94.8% (-3.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.6% → 91.1% (-1.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,001,084 | 1,532,517 | +468,567 | +30.57% | skill 13/13; base 13/13 |
| claude-code | management-eval-001-mcp-readonly | 156,154 | 35,209 | +120,945 | +343.51% | skill 1/1; base 1/1 |
| claude-code | management-eval-002-validate-422 | 114,465 | 142,175 | -27,710 | -19.49% | skill 1/1; base 1/1 |
| claude-code | management-eval-003-reset-is-destructive | 150,422 | 142,182 | +8,240 | +5.80% | skill 1/1; base 1/1 |
| claude-code | management-eval-004-token-acts-as-owner | 185,798 | 142,045 | +43,753 | +30.80% | skill 1/1; base 1/1 |
| claude-code | management-eval-005-import-replace | 150,021 | 142,673 | +7,348 | +5.15% | skill 1/1; base 1/1 |
| claude-code | management-eval-006-lineage-semantic-layer | 231,997 | 106,406 | +125,591 | +118.03% | skill 1/1; base 1/1 |
| claude-code | management-eval-007-patch-vs-put-sql-attribute | 113,493 | 142,102 | -28,609 | -20.13% | skill 1/1; base 1/1 |
| claude-code | management-eval-008-run-grain-measure | 113,035 | 107,414 | +5,621 | +5.23% | skill 1/1; base 1/1 |
| claude-code | management-eval-009-relationship-evidence | 200,647 | 142,446 | +58,201 | +40.86% | skill 1/1; base 1/1 |
| claude-code | management-eval-010-import-exact-readback | 112,878 | 107,031 | +5,847 | +5.46% | skill 1/1; base 1/1 |
| claude-code | management-eval-011-result-publication-mode | 157,275 | 107,974 | +49,301 | +45.66% | skill 1/1; base 1/1 |
| claude-code | management-eval-012-promotion-readback-failure | 113,036 | 107,085 | +5,951 | +5.56% | skill 1/1; base 1/1 |
| claude-code | management-eval-013-composed-lifecycle | 201,863 | 107,775 | +94,088 | +87.30% | skill 1/1; base 1/1 |
| codex | All cases | 957,349 | 595,372 | +361,977 | +60.80% | skill 13/13; base 13/13 |
| codex | management-eval-001-mcp-readonly | 77,860 | 55,234 | +22,626 | +40.96% | skill 1/1; base 1/1 |
| codex | management-eval-002-validate-422 | 77,976 | 60,038 | +17,938 | +29.88% | skill 1/1; base 1/1 |
| codex | management-eval-003-reset-is-destructive | 48,245 | 46,350 | +1,895 | +4.09% | skill 1/1; base 1/1 |
| codex | management-eval-004-token-acts-as-owner | 44,576 | 40,432 | +4,144 | +10.25% | skill 1/1; base 1/1 |
| codex | management-eval-005-import-replace | 52,637 | 40,956 | +11,681 | +28.52% | skill 1/1; base 1/1 |
| codex | management-eval-006-lineage-semantic-layer | 109,088 | 40,542 | +68,546 | +169.07% | skill 1/1; base 1/1 |
| codex | management-eval-007-patch-vs-put-sql-attribute | 48,858 | 48,516 | +342 | +0.70% | skill 1/1; base 1/1 |
| codex | management-eval-008-run-grain-measure | 44,166 | 13,486 | +30,680 | +227.50% | skill 1/1; base 1/1 |
| codex | management-eval-009-relationship-evidence | 45,358 | 40,944 | +4,414 | +10.78% | skill 1/1; base 1/1 |
| codex | management-eval-010-import-exact-readback | 49,102 | 44,884 | +4,218 | +9.40% | skill 1/1; base 1/1 |
| codex | management-eval-011-result-publication-mode | 44,117 | 40,570 | +3,547 | +8.74% | skill 1/1; base 1/1 |
| codex | management-eval-012-promotion-readback-failure | 44,728 | 54,407 | -9,679 | -17.79% | skill 1/1; base 1/1 |
| codex | management-eval-013-composed-lifecycle | 270,638 | 69,013 | +201,625 | +292.16% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,958,433 | 2,127,889 | +830,544 | +39.03% | skill 26/26; base 26/26 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 5 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/write-api.md:
  "## Bulk import / export" in SKILL.md (lines 97-109)
  vs "## Model interchange" in references/write-api.md (lines 59-68) (`SKILL.md:97`)
- **MEDIUM** QUALITY/quality_correctness: Resource-based skill lacks documentation of available resources (`skills/nvidia-ontology-management/SKILL.md`)
- **MEDIUM** QUALITY/quality_reliability: MCP skill lacks connection/error guidance (`skills/nvidia-ontology-management/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in modeling.md (`skills/nvidia-ontology-management/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/nvidia-ontology-management/SKILL.md`)
- 1 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
