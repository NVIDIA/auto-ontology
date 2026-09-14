# Skill Card

## Description

nvidia-ontology-install installs and troubleshoots a GSF (Generative Semantic Fabric) deployment for developers bringing the stack up, or connecting MCP to an instance that is already running.

This skill is for research and development until a catalog eval run is recorded below.

## Owner

NVIDIA Ontology Team

## License/Terms of Use

Apache 2.0

## Use Case

Developers setting up GSF locally (Docker Compose, `--dev`, `--ds`) or via Helm, and agents that must recover from bring-up failures without mining web docs.

## Deployment Geography for Use

Global

### Requirements / Dependencies

Requires API Key or External Credential: Yes
Credential Type(s): NVIDIA NIM API key (`DEFAULT_MODELS_API_KEY`); optional source-database connection strings; GitHub credentials to `uvx` gsf-mcp while GSF is unpublished on PyPI

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. Install logging redacts passwords, tokens, and API keys.

## Known Risks and Mitigations

Risk: The skill may propose compose, helm, or `.env` edits that break a running instance or store connection passwords in plaintext when Vault is only partially configured.
Mitigation: Ask the required target questions first; wrap commands with `scripts/log_install.sh`; treat partial Vault config as plaintext; do not invent a second installer.

Risk: Guidance could be copied into the public nvidia/skills catalog.
Mitigation: This skill contains no staging hostnames, cluster names, or Vault paths.

## References

- Repository-root `README.md`, `.env.example`, `dev_tools/setup_env.sh`, `DEPLOYMENT.md`
- `docs/mcp.md` troubleshooting
- [evals/evals.json](evals/evals.json) — candidate dataset; no `BENCHMARK.md` until a Tier-3 eval runs

## Skill Output

Output type(s): Shell commands, configuration instructions, install log lines
Output format: Markdown with inline bash code blocks; append-only `.nvidia-ontology-install.log`
Output parameters: Commands run from the GSF repository root unless noted
Other properties: `.nvidia-ontology-install.log` is gitignored; do not commit it

## Skill Version

0.1.0 (source: frontmatter)

## Evaluation Agents Used

Not yet run.

## Evaluation Tasks

7 candidate tasks in [evals/evals.json](evals/evals.json). No isolated-pod eval has been executed.

## Evaluation Results

Not yet evaluated. Do not invent uplift numbers.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility. When used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case. Report quality, risk, or security concerns through NVIDIA's published vulnerability process.

Do not fabricate `skill.oms.sig`. Catalog signing is a follow-up with the nvidia/skills pipeline.
