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
Credential Type(s): NVIDIA NIM API key (`DEFAULT_MODELS_API_KEY`); optional source-database connection strings; GitHub credentials to `uvx` auto-ontology-mcp while Auto Ontology is unpublished on PyPI

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

SkillEvaluator 0.2.1 with OpenCode and `switchyard/openai/gpt-5.6-sol` in Docker isolation. One attempt per case and condition used the same prompt, model, grader, and environment. The agent route used OpenAI-compatible chat completions because the endpoint's Responses route failed encrypted-content affinity.

## Evaluation Tasks

All 7 cases in `evals/evals.json`.

## Evaluation Results

Full one-attempt Tier-3 matrix:

| Metric | With skill | Baseline | Delta |
| --- | ---: | ---: | ---: |
| Pass@1 | 1.000 | 0.286 | +0.714 |
| Security | 1.000 | 1.000 | 0.000 |
| Skill execution | 0.991 | 0.536 | +0.455 |
| Skill efficiency | 0.805 | 0.106 | +0.699 |
| Accuracy | 0.857 | 0.543 | +0.314 |
| Goal accuracy | 0.747 | 0.336 | +0.411 |
| Behavior adherence | 0.869 | 0.393 | +0.476 |
| Overall | 0.878 | 0.486 | **+0.393** |

Provider and container failures initially left trials unscored. Those case-condition pairs were rerun with the same configuration at lower concurrency and combined with the successful attempts. The matrix demonstrates broad positive lift, but repeated attempts remain pending.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility. When used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case. Report quality, risk, or security concerns through NVIDIA's published vulnerability process.

Do not fabricate `skill.oms.sig`. Catalog signing is a follow-up with the nvidia/skills pipeline.
