# Skill Benchmark: nvidia-ontology-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-ontology-setup`
- Validation status: `passed`
- Evaluation date: 2026-10-29
- Evaluator version: `1.6.4`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 10 evaluation tasks (10 positive)
- Dataset digest: `sha256:dcbcf0848aefb19cd5b010e9a0d69ed8920e61dc3fdba560ca0f88cd48a8da96` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvidia-ontology-setup | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nvidia-ontology-setup | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nvidia-ontology-setup | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nvidia-ontology-setup | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nvidia-ontology-setup | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |

- Controls for `nvidia-ontology-setup`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 92.4% — baseline ran, but no comparable score was available; uplift unavailable | 90.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 80.0% → 100.0% (+20.0 points) | 85.0% → 100.0% (+15.0 points) |
| Correctness | 54.0% → 100.0% (+46.0 points) | 60.0% → 98.0% (+38.0 points) |
| Discoverability | 88.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 45.8% → 95.0% (+49.2 points) | 65.2% → 83.8% (+18.6 points) |
| Efficiency | 79.1% — baseline ran, but no comparable score was available; uplift unavailable | 81.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.7% → 96.4% (-1.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.0% → 90.3% (+1.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,069,329 | 1,420,288 | -350,959 | -24.71% | skill 10/10; base 10/10 |
| claude-code | setup-eval-001-underspecified | 116,140 | 107,306 | +8,834 | +8.23% | skill 1/1; base 1/1 |
| claude-code | setup-eval-002-use-existing-scripts | 75,563 | 269,555 | -193,992 | -71.97% | skill 1/1; base 1/1 |
| claude-code | setup-eval-003-auth-secret-cookies | 117,098 | 106,927 | +10,171 | +9.51% | skill 1/1; base 1/1 |
| claude-code | setup-eval-004-postgres-port-host-only | 71,061 | 106,594 | -35,533 | -33.33% | skill 1/1; base 1/1 |
| claude-code | setup-eval-005-mcp-already-up | 115,185 | 143,797 | -28,612 | -19.90% | skill 1/1; base 1/1 |
| claude-code | setup-eval-006-vault-partial | 107,609 | 107,229 | +380 | +0.35% | skill 1/1; base 1/1 |
| claude-code | setup-eval-007-embed-triplet-mismatch | 122,771 | 106,963 | +15,808 | +14.78% | skill 1/1; base 1/1 |
| claude-code | setup-eval-008-connect-source-workflow | 116,659 | 220,389 | -103,730 | -47.07% | skill 1/1; base 1/1 |
| claude-code | setup-eval-009-env-vs-ui-connections | 110,852 | 143,445 | -32,593 | -22.72% | skill 1/1; base 1/1 |
| claude-code | setup-eval-010-helm-install | 116,391 | 108,083 | +8,308 | +7.69% | skill 1/1; base 1/1 |
| codex | All cases | 515,404 | 630,968 | -115,564 | -18.32% | skill 10/10; base 10/10 |
| codex | setup-eval-001-underspecified | 29,577 | 78,029 | -48,452 | -62.09% | skill 1/1; base 1/1 |
| codex | setup-eval-002-use-existing-scripts | 42,276 | 56,252 | -13,976 | -24.85% | skill 1/1; base 1/1 |
| codex | setup-eval-003-auth-secret-cookies | 46,848 | 45,371 | +1,477 | +3.26% | skill 1/1; base 1/1 |
| codex | setup-eval-004-postgres-port-host-only | 33,887 | 44,980 | -11,093 | -24.66% | skill 1/1; base 1/1 |
| codex | setup-eval-005-mcp-already-up | 50,996 | 51,757 | -761 | -1.47% | skill 1/1; base 1/1 |
| codex | setup-eval-006-vault-partial | 78,819 | 48,898 | +29,921 | +61.19% | skill 1/1; base 1/1 |
| codex | setup-eval-007-embed-triplet-mismatch | 67,006 | 62,269 | +4,737 | +7.61% | skill 1/1; base 1/1 |
| codex | setup-eval-008-connect-source-workflow | 63,597 | 88,752 | -25,155 | -28.34% | skill 1/1; base 1/1 |
| codex | setup-eval-009-env-vs-ui-connections | 56,241 | 82,243 | -26,002 | -31.62% | skill 1/1; base 1/1 |
| codex | setup-eval-010-helm-install | 46,157 | 72,417 | -26,260 | -36.26% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,584,733 | 2,051,256 | -466,523 | -22.74% | skill 20/20; base 20/20 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 6 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Resource-based skill lacks documentation of available resources (`skills/nvidia-ontology-setup/SKILL.md`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: Never ask the user (`SKILL.md:62`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: curl -s -o /dev/null -w '%{http_code}\n' "$AUTO_ONTOLOGY_API_URL/.well-known/oauth-authorization-server"
```

`200` is required. Anything else: run the frontend from the checkout
(`pnpm dev`) and poin (`SKILL.md:200`)
- **LOW** QUALITY/quality_discoverability: Skill uses exclusivity language that conflicts with composability (`skills/nvidia-ontology-setup/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/nvidia-ontology-setup/SKILL.md`)
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
