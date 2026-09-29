# Skill Card

## Description

nvidia-ontology-management inspects, models, manages, and safely publishes through the current Auto Ontology semantic layer, including source-grounded concepts, relationships, measures, model import/export, governed result definitions, exact readback, and rollback boundaries.

This skill is for research and development. The representative pilot below is not full catalog certification.

## Owner

Auto Ontology Team

## License/Terms of Use

Apache 2.0

## Use Case

Agents that must understand or change ontology meaning, define reusable measures and relationships, import/export a model safely, or publish a supported governed definition or result handoff without confusing request success with certification.

## Deployment Geography for Use

Global

### Requirements / Dependencies

Requires API Key or External Credential: Yes
Credential Type(s): Auto Ontology API token (`x-api-key`) or signed-in session / SSO bearer with `catalog:edit` (and `modelInterchange:*` / `semanticCompilation:manage` for those operations)

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate.

## Known Risks and Mitigations

Risk: The skill may apply catalog writes or a compile reset that destroys the compiled semantic layer or replaces a model on import (`replace` defaults to true).
Mitigation: MCP is read-only; confirm writes with the user; never treat `POST /api/semantic-compilation/reset` as cleanup; confirm before import with `replace=true`.

Risk: An API token acts as its owner; a leaked admin token can edit the catalog.
Mitigation: State that tokens inherit the owner's role; prefer least-privilege owners; revocation is immediate in the UI.

Risk: A successful import or non-empty query may conceal stale properties, retained relationships, wrong grain, or an unsupported publication claim.
Mitigation: Require a scoped backup, isolated application, exact re-export comparison, positive and negative query probes, and receipt-only behavior when no approved writer exists.

## References

- `docs/openapi/auto-ontology-api.json` (operation ids and `x-auto-ontology-permissions`)
- `mcp/auto_ontology_mcp/tools.py` (read-only MCP allow-list)
- [assets/runtime-contract.yaml](assets/runtime-contract.yaml)
- [references/write-api.md](references/write-api.md)
- [references/modeling.md](references/modeling.md)
- [references/publication.md](references/publication.md)
- [evals/evals.json](evals/evals.json) — candidate dataset; representative Tier-3 result below

## Skill Output

Output type(s): Semantic specifications, API calls, proposed catalog edits, publication and rollback receipts, verification questions
Output format: HTTP against the Next.js `/api` gateway; approved source-writer handoffs; Markdown explanations
Output parameters: Permission tags as in OpenAPI; source and semantic IDs; no credential or raw-data dumps
Other properties: Does not modify MCP; unsupported writes stop at an explicit receipt

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
| Pass@1 | 1.000 | 0.462 | +0.538 |
| Security | 0.923 | 1.000 | -0.077 |
| Skill execution | 0.962 | 0.481 | +0.481 |
| Skill efficiency | 0.817 | 0.117 | +0.700 |
| Accuracy | 0.831 | 0.508 | +0.323 |
| Goal accuracy | 0.656 | 0.389 | +0.268 |
| Behavior adherence | 0.750 | 0.397 | +0.353 |
| Overall | 0.823 | 0.482 | **+0.341** |

Provider and container failures initially left trials unscored. Those case-condition pairs were rerun with the same configuration at lower concurrency and combined with the successful attempts. Localhost read-only probes account for the small security delta. The matrix demonstrates broad positive lift, but repeated attempts and a fixture-backed live lifecycle remain pending.

## Ethical Considerations

Ontology edits change how questions resolve to SQL. Human review is expected before applying import/replace or compile reset. Report quality, risk, or security concerns through NVIDIA's published vulnerability process.

Do not fabricate `skill.oms.sig`. Catalog signing is a follow-up with the nvidia/skills pipeline.
