# Skill Card

## Description

nvidia-ontology inspects, models, manages, and safely publishes through the current GSF semantic layer, including source-grounded concepts, relationships, measures, model import/export, governed result definitions, exact readback, and rollback boundaries.

This skill is for research and development until a catalog eval run is recorded below.

## Owner

NVIDIA Ontology Team

## License/Terms of Use

Apache 2.0

## Use Case

Agents that must understand or change ontology meaning, define reusable measures and relationships, import/export a model safely, or publish a supported governed definition or result handoff without confusing request success with certification.

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

Risk: A successful import or non-empty query may conceal stale properties, retained relationships, wrong grain, or an unsupported publication claim.
Mitigation: Require a scoped backup, isolated application, exact re-export comparison, positive and negative query probes, and receipt-only behavior when no approved writer exists.

## References

- `docs/openapi/gsf-api.json` (operation ids and `x-gsf-permissions`)
- `mcp/gsf_mcp/tools.py` (read-only MCP allow-list)
- [runtime-contract.yaml](runtime-contract.yaml)
- [references/write-api.md](references/write-api.md)
- [references/modeling.md](references/modeling.md)
- [references/publication.md](references/publication.md)
- [evals/evals.json](evals/evals.json) — candidate dataset; no `BENCHMARK.md` until a Tier-3 eval runs

## Skill Output

Output type(s): Semantic specifications, API calls, proposed catalog edits, publication and rollback receipts, verification questions
Output format: HTTP against the Next.js `/api` gateway; approved source-writer handoffs; Markdown explanations
Output parameters: Permission tags as in OpenAPI; source and semantic IDs; no credential or raw-data dumps
Other properties: Does not modify MCP; unsupported writes stop at an explicit receipt

## Skill Version

0.2.0 (source: frontmatter)

## Evaluation Agents Used

Not yet run.

## Evaluation Tasks

13 candidate tasks in [evals/evals.json](evals/evals.json). No isolated-pod eval has been executed.

## Evaluation Results

Not yet evaluated. Do not invent uplift numbers.

## Ethical Considerations

Ontology edits change how questions resolve to SQL. Human review is expected before applying import/replace or compile reset. Report quality, risk, or security concerns through NVIDIA's published vulnerability process.

Do not fabricate `skill.oms.sig`. Catalog signing is a follow-up with the nvidia/skills pipeline.
