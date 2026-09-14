from __future__ import annotations

from gsf.utils.llm_invoke import safe_invoke_text

_MERGE_PROMPT = """\
You are refining a database question that already incorporates previous clarifications.

Current question (do NOT discard any formula, definition, or constraint it contains):
{current_question}

New clarification:
Q: {new_q}
A: {new_a}

External knowledge (for context only — already passed to SQL generation separately; \
use it to avoid coining pseudo-column names):
{relevant_knowledge}

Rewrite the current question to incorporate the new clarification. Rules:
- Preserve every formula, definition, constraint, and column name already in the current question, \
  unless explicitly contradicted by the new answer, in which case you may replace misleading information.
- If the new clarification defines or renames a metric, use that name consistently everywhere \
  in the rewritten question — including in aggregations (average, median, count) that reference it.
- If the answer provides an EXPLICIT formula (exact operator, exact column names, exact constants), \
  embed it using the user's exact natural-language terms and phrasing — preserve the formula \
  structure but write variable names as natural-language phrases, not as snake_case or SQL \
  function calls (e.g. write "total point count, defaulting to 1,000,000 when unavailable" \
  rather than "COALESCE(total_point_count, 1000000)").
- If the user explicitly names a field or column identifier in parentheses, \
  preserve that identifier name exactly — do not paraphrase or drop it.
- If the answer describes a calculation vaguely (no exact operator or constants), or indicates a misunderstanding has occured,\
  reflect the description using the user's words — do NOT invent a specific formula or add a guessed example.
- Do NOT introduce external definitions, formulas, or example expressions beyond what the user \
  explicitly stated in the answer above.
- Do NOT invent column names. If a metric is a computed expression (e.g. defined in external \
  knowledge), refer to it by its formula or its KB name, not a made-up column name.
- Output only the rewritten question, nothing else."""


def merge_clarification(
    current_question: str,
    new_turn: dict,
    llm,
    relevant_knowledge: str = "",
) -> str:
    """Incrementally refine current_question with one new Q&A turn."""
    prompt = _MERGE_PROMPT.format(
        current_question=current_question,
        new_q=new_turn["q"],
        new_a=new_turn["a"],
        relevant_knowledge=relevant_knowledge or "None",
    )
    return safe_invoke_text(llm, prompt).strip()
