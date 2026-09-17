# Skill Card

## Description

nvidia-ontology-setup configures and troubleshoots an Auto Ontology deployment for developers bringing the stack up, connecting a source, or connecting MCP to an instance that is already running.

This skill is for research and development. The representative pilot below is not full catalog certification.

## Owner

Auto Ontology Team

## License/Terms of Use

Apache 2.0

## Use Case

Developers setting up Auto Ontology locally (Docker Compose, `--dev`, `--ds`) or via Helm, and agents that must recover from bring-up failures without mining web docs.

## Deployment Geography for Use

Global

### Requirements / Dependencies

Requires API Key or External Credential: Yes
Credential Type(s): NVIDIA NIM API key (`DEFAULT_MODELS_API_KEY`); optional source-database connection strings; GitHub credentials to `uvx` gsf-mcp while Auto Ontology is unpublished on PyPI

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. Install logging redacts passwords, tokens, and API keys.

## Known Risks and Mitigations

Risk: The skill may propose compose, helm, or `.env` edits that break a running instance or store connection passwords in plaintext when Vault is only partially configured.
Mitigation: Ask the required target questions first; wrap commands with `scripts/log_setup.sh`; treat partial Vault config as plaintext; do not invent a second installer.

Risk: Guidance could be copied into the public nvidia/skills catalog.
Mitigation: This skill contains no staging hostnames, cluster names, or Vault paths.

## References

- Repository-root `README.md`, `.env.example`, `dev_tools/setup_env.sh`, `DEPLOYMENT.md`
- `docs/mcp.md` troubleshooting
- [assets/runtime-contract.yaml](assets/runtime-contract.yaml)
- [evals/evals.json](evals/evals.json) — candidate dataset; representative Tier-3 result below

## Skill Output

Output type(s): Deployment plans, shell commands, configuration instructions, install log lines, connection and readiness receipts
Output format: Markdown with inline bash code blocks; append-only `.nvidia-ontology-setup.log`
Output parameters: Commands run from the Auto Ontology repository root unless noted
Other properties: `.nvidia-ontology-setup.log` is gitignored; do not commit it

## Skill Version

0.2.0 (source: frontmatter)

## Evaluation Agents Used

SkillEvaluator 0.2.1 with OpenCode and `switchyard/openai/gpt-5.6-sol` in Docker isolation. One with-skill and one baseline attempt used the same prompt, model, grader, and environment. The agent route used OpenAI-compatible chat completions because the endpoint's Responses route failed encrypted-content affinity.

## Evaluation Tasks

One representative case from 7 candidates: `setup-eval-001-underspecified`.

## Evaluation Results

Representative Tier-3 pilot:

| Metric | With skill | Baseline | Delta |
| --- | ---: | ---: | ---: |
| Pass@1 | 1.00 | 0.00 | +1.00 |
| Security | 1.00 | 1.00 | 0.00 |
| Skill execution | 1.00 | 0.50 | +0.50 |
| Skill efficiency | 1.00 | 0.00 | +1.00 |
| Accuracy | 1.00 | 0.60 | +0.40 |
| Goal accuracy | 1.00 | 0.20 | +0.80 |
| Behavior adherence | 1.00 | 0.25 | +0.75 |
| Overall | 1.00 | 0.425 | **+0.575** |

This one-attempt smoke test demonstrates activation and behavior attribution, not statistical reliability. The full 7-case matrix and repeated attempts remain pending.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility. When used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case. Report quality, risk, or security concerns through NVIDIA's published vulnerability process.

Do not fabricate `skill.oms.sig`. Catalog signing is a follow-up with the nvidia/skills pipeline.
