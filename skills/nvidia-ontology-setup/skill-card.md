## Description: <br>
Set up or troubleshoot the Auto Ontology runtime, covering Helm (the official install), Docker Compose, developer setup, and MCP connection to an existing deployment. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and platform engineers use this skill to install Auto Ontology with the published Helm chart or Docker Compose, run developer workflows, connect source databases, troubleshoot setup failures, and connect an MCP client to an existing deployment. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, OAuth Token] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Auto Ontology setup troubleshooting](references/troubleshooting.md) <br>
- [Runtime contract](assets/runtime-contract.yaml) <br>
- [Table-level scope guarantee (issue #255)](https://github.com/NVIDIA/auto-ontology/issues/255) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, API Calls] <br>
**Output Format:** [Markdown with inline bash and YAML code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
10 evaluation tasks (10 positive), 1 attempt per task, each run in an isolated sandbox pod; baseline is the same task attempted without the skill. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? <br>
- Correctness: Is the answer correct? <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? (equal-weight goal completion and expected workflow adherence) <br>
- Efficiency: Did it avoid wasted tool calls and token usage? (50% tool-call productivity, 50% token efficiency) <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.4% (uplift unavailable) | 90.2% (uplift unavailable) |
| Security | 95.0% → 100.0% (+5.0 points) | 80.0% → 100.0% (+20.0 points) |
| Correctness | 48.0% → 100.0% (+52.0 points) | 60.0% → 96.0% (+36.0 points) |
| Discoverability | 94.0% (uplift unavailable) | 93.5% (uplift unavailable) |
| Effectiveness | 41.4% → 95.0% (+53.6 points) | 59.1% → 79.7% (+20.6 points) |
| Efficiency | 83.1% (uplift unavailable) | 82.0% (uplift unavailable) |

## Skill Version(s): <br>
0.3.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
