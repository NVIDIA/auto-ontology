# Skill Card

## Description

gsf-ontology inspects and modifies the GSF semantic layer (glossary terms, SQL attributes, lineage, model import/export, compilation) for developers and partners who need to change ontology meaning.

This skill is for research and development until a catalog eval run is recorded below.

## Owner

NVIDIA GSF Team

## License/Terms of Use

Apache 2.0

## Use Case

Agents that must edit GSF terms or SQL attributes, import/export model YAML, or explain dataset meaning via the semantic layer — not via raw database schemas.

## Deployment Geography for Use

Global

### Requirements / Dependencies

Requires API Key or External Credential: Yes
Credential Type(s): GSF API token (`x-api-key`) or signed-in session / SSO bearer with `catalog:edit` (and `modelInterchange:*` / `semanticCompilation:manage` for those operations)

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate.

## Known Risks and Mitigations

Risk: The skill may apply catalog writes or a compile reset that destroys the compiled semantic layer or replaces a model on import (`replace` defaults to true).
Mitigation: MCP is read-only; confirm writes with the user; never treat `POST /api/semantic-compilation/reset` as cleanup; confirm before import with `replace=true`.

Risk: An API token acts as its owner; a leaked admin token can edit the catalog.
Mitigation: State that tokens inherit the owner's role; prefer least-privilege owners; revocation is immediate in the UI.

## References

- `docs/openapi/gsf-api.json` (operation ids and `x-gsf-permissions`)
- `mcp/gsf_mcp/tools.py` (read-only MCP allow-list)
- [references/write-api.md](references/write-api.md)
- [evals/evals.json](evals/evals.json) — candidate dataset; no `BENCHMARK.md` until a Tier-3 eval runs

## Skill Output

Output type(s): API calls, proposed catalog edits, verification questions
Output format: HTTP against the Next.js `/api` gateway; Markdown explanations
Output parameters: Permission tags as in OpenAPI; no schema dumps
Other properties: Does not modify MCP; does not browse raw schemas for meaning

## Skill Version

0.1.0 (source: frontmatter)

## Evaluation Agents Used

Not yet run.

## Evaluation Tasks

7 candidate tasks in [evals/evals.json](evals/evals.json). No isolated-pod eval has been executed.

## Evaluation Results

Not yet evaluated. Do not invent uplift numbers.

## Ethical Considerations

Ontology edits change how questions resolve to SQL. Human review is expected before applying import/replace or compile reset. Report quality, risk, or security concerns through NVIDIA's published vulnerability process.

Do not fabricate `skill.oms.sig`. Catalog signing is a follow-up with the nvidia/skills pipeline.
