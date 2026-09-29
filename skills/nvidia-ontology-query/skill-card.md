# Skill Card

## Description

nvidia-ontology-query guides grounded queries against the current Auto Ontology implementation: MCP-first reads, REST fallback, semantic discovery, authentication, conversations, and independent validation of generated SQL, returned rows, truncation, and answer prose.

This skill is for research and development. The representative pilot below is not full catalog certification.

## Owner

Auto Ontology Team

## License/Terms of Use

Apache 2.0

## Use Case

Partners embedding Auto Ontology in an agent harness, AI-Q, NemoClaw, or another app; agents that must discover ontology meaning, ask grounded descriptive or diagnostic questions, and validate the resulting claims.

## Deployment Geography for Use

Global

### Requirements / Dependencies

Requires API Key or External Credential: Yes
Credential Type(s): MCP user login against Auto Ontology; or Auto Ontology API token; or SSO id token (AI-Q). Optional NVIDIA NIM key is a deployment concern (`nvidia-ontology-setup`), not this skill.

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate.

## Known Risks and Mitigations

Risk: An agent may call FastAPI `:3001` with `x-auto-ontology-user-id` and impersonate a user, or share `conversation_id` across users.
Mitigation: Skill requires the Next.js gateway for public clients; FastAPI is an internal boundary; conversation ids are per Auto Ontology user.

Risk: SSO bearer verification does not check `aud`, so any token from the configured issuer is accepted.
Mitigation: Document the intended AI-Q model; map to an existing Auto Ontology user; do not configure a deployment-wide MCP token.

Risk: Executable SQL may still answer the wrong population or multiply a measure, while fluent prose may omit or contradict returned values.
Mitigation: Resolve the material question first, then validate SQL, rows, truncation, and prose independently and fail closed on material mismatch.

## References

- `mcp/README.md`, `docs/mcp.md`, `mcp/auto_ontology_mcp/tools.py`
- `docs/openapi/auto-ontology-api.json`
- `frontend/auth/resolve-user.ts`, `frontend/auth/bearer.ts`
- [assets/runtime-contract.yaml](assets/runtime-contract.yaml)
- [references/stack.md](references/stack.md)
- [references/query-validation.md](references/query-validation.md)
- [evals/evals.json](evals/evals.json) — candidate dataset; representative Tier-3 result below

## Skill Output

Output type(s): MCP tool calls, HTTP API calls, grounded query receipts, validated descriptive or diagnostic answers
Output format: Markdown with a small Python `requests` example (from the Auto Ontology README)
Output parameters: Question contract, SQL, rows, row count, truncation, answer, and gaps; no credentials
Other properties: Unopinionated scaffolding; MCP tool lists are not copied; execution success is not semantic validation

## Skill Version

0.2.0 (source: frontmatter)

## Evaluation Agents Used

SkillEvaluator 0.2.1 with OpenCode and `switchyard/openai/gpt-5.6-sol` in Docker isolation. One attempt per case and condition used the same prompt, model, grader, and environment. The agent route used OpenAI-compatible chat completions because the endpoint's Responses route failed encrypted-content affinity.

## Evaluation Tasks

All 13 cases in `evals/evals.json`.

## Evaluation Results

Full one-attempt Tier-3 matrix:

| Metric | With skill | Baseline | Delta |
| --- | ---: | ---: | ---: |
| Pass@1 | 0.769 | 0.538 | +0.231 |
| Security | 1.000 | 1.000 | 0.000 |
| Skill execution | 0.692 | 0.365 | +0.327 |
| Skill efficiency | 0.609 | 0.087 | +0.522 |
| Accuracy | 0.723 | 0.538 | +0.185 |
| Goal accuracy | 0.600 | 0.400 | +0.200 |
| Behavior adherence | 0.712 | 0.513 | +0.199 |
| Overall | 0.723 | 0.484 | **+0.239** |

Provider and container failures initially left trials unscored. Those case-condition pairs were rerun with the same configuration at lower concurrency and combined with the successful attempts. The matrix demonstrates positive lift, but repeated attempts remain pending.

## Ethical Considerations

Auto Ontology answers are only as correct as the compiled ontology and the caller's permissions. Do not bypass the semantic layer to dump raw schemas. Report quality, risk, or security concerns through NVIDIA's published vulnerability process.

Do not fabricate `skill.oms.sig`. Catalog signing is a follow-up with the nvidia/skills pipeline.
