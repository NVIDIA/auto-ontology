# Skill Card

## Description

nvidia-ontology-agent guides an agent calling the current GSF implementation: MCP-first reads, REST fallback, semantic discovery, authentication, conversations, and independent validation of generated SQL, returned rows, truncation, and answer prose.

This skill is for research and development until a catalog eval run is recorded below.

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
- [runtime-contract.yaml](runtime-contract.yaml)
- [references/stack.md](references/stack.md)
- [references/query-validation.md](references/query-validation.md)
- [evals/evals.json](evals/evals.json) — candidate dataset; no `BENCHMARK.md` until a Tier-3 eval runs

## Skill Output

Output type(s): MCP tool calls, HTTP API calls, grounded query receipts, validated descriptive or diagnostic answers
Output format: Markdown with a small Python `requests` example (from the GSF README)
Output parameters: Question contract, SQL, rows, row count, truncation, answer, and gaps; no credentials
Other properties: Unopinionated scaffolding; MCP tool lists are not copied; execution success is not semantic validation

## Skill Version

0.2.0 (source: frontmatter)

## Evaluation Agents Used

Not yet run.

## Evaluation Tasks

13 candidate tasks in [evals/evals.json](evals/evals.json). No isolated-pod eval has been executed.

## Evaluation Results

Not yet evaluated. Do not invent uplift numbers.

## Ethical Considerations

GSF answers are only as correct as the compiled ontology and the caller's permissions. Do not bypass the semantic layer to dump raw schemas. Report quality, risk, or security concerns through NVIDIA's published vulnerability process.

Do not fabricate `skill.oms.sig`. Catalog signing is a follow-up with the nvidia/skills pipeline.
