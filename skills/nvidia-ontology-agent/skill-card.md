# Skill Card

## Description

nvidia-ontology-agent guides an agent calling the current GSF implementation: MCP-first reads, REST fallback, semantic discovery, authentication, conversations, and independent validation of generated SQL, returned rows, truncation, and answer prose.

This skill is for research and development. The representative pilot below is not full catalog certification.

## Owner

NVIDIA Ontology Team

## License/Terms of Use

Apache 2.0

## Use Case

Partners embedding GSF in an agent harness, AI-Q, NemoClaw, or another app; agents that must discover ontology meaning, ask grounded descriptive or diagnostic questions, and validate the resulting claims.

## Deployment Geography for Use

Global

### Requirements / Dependencies

Requires API Key or External Credential: Yes
Credential Type(s): MCP user login against GSF; or GSF API token; or SSO id token (AI-Q). Optional NVIDIA NIM key is a deployment concern (`nvidia-ontology-install`), not this skill.

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate.

## Known Risks and Mitigations

Risk: An agent may call FastAPI `:3001` with `x-gsf-user-id` and impersonate a user, or share `conversation_id` across users.
Mitigation: Skill requires the Next.js gateway for public clients; FastAPI is an internal boundary; conversation ids are per GSF user.

Risk: SSO bearer verification does not check `aud`, so any token from the configured issuer is accepted.
Mitigation: Document the intended AI-Q model; map to an existing GSF user; do not configure a deployment-wide MCP token.

Risk: Executable SQL may still answer the wrong population or multiply a measure, while fluent prose may omit or contradict returned values.
Mitigation: Resolve the material question first, then validate SQL, rows, truncation, and prose independently and fail closed on material mismatch.

## References

- `mcp/README.md`, `docs/mcp.md`, `mcp/gsf_mcp/tools.py`
- `docs/openapi/gsf-api.json`
- `frontend/auth/resolve-user.ts`, `frontend/auth/bearer.ts`
- [assets/runtime-contract.yaml](assets/runtime-contract.yaml)
- [references/stack.md](references/stack.md)
- [references/query-validation.md](references/query-validation.md)
- [evals/evals.json](evals/evals.json) — candidate dataset; representative Tier-3 result below

## Skill Output

Output type(s): MCP tool calls, HTTP API calls, grounded query receipts, validated descriptive or diagnostic answers
Output format: Markdown with a small Python `requests` example (from the GSF README)
Output parameters: Question contract, SQL, rows, row count, truncation, answer, and gaps; no credentials
Other properties: Unopinionated scaffolding; MCP tool lists are not copied; execution success is not semantic validation

## Skill Version

0.2.0 (source: frontmatter)

## Evaluation Agents Used

SkillEvaluator 0.2.1 with OpenCode and `switchyard/openai/gpt-5.6-sol` in Docker isolation. One with-skill and one baseline attempt used the same prompt, model, grader, and environment. The agent route used OpenAI-compatible chat completions because the endpoint's Responses route failed encrypted-content affinity.

## Evaluation Tasks

One representative case from 13 candidates: `agent-eval-009-run-metric-fanout`.

## Evaluation Results

Representative Tier-3 pilot:

| Metric | With skill | Baseline | Delta |
| --- | ---: | ---: | ---: |
| Pass@1 | 1.00 | 1.00 | 0.00 |
| Security | 1.00 | 1.00 | 0.00 |
| Skill execution | 1.00 | 0.00 | +1.00 |
| Skill efficiency | 1.00 | 0.00 | +1.00 |
| Accuracy | 1.00 | 1.00 | 0.00 |
| Goal accuracy | 1.00 | 0.95 | +0.05 |
| Behavior adherence | 1.00 | 0.50 | +0.50 |
| Overall | 1.00 | 0.575 | **+0.425** |

This one-attempt smoke test demonstrates activation and behavior attribution, not statistical reliability. The full 13-case matrix and repeated attempts remain pending.

## Ethical Considerations

GSF answers are only as correct as the compiled ontology and the caller's permissions. Do not bypass the semantic layer to dump raw schemas. Report quality, risk, or security concerns through NVIDIA's published vulnerability process.

Do not fabricate `skill.oms.sig`. Catalog signing is a follow-up with the nvidia/skills pipeline.
