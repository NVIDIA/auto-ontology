"""External-knowledge (KB) parsing, formatting, and coverage matching.

Split out of clarify.py: this module owns the "does external knowledge cover
this term" concern — parsing the formatted_kb string into entries, slimming
it for cheap coverage prompts, and expanding parent entries with their
children. Entity extraction / VDB resolution lives in entity_resolution.py;
clarification-turn orchestration lives in clarify.py.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata

import requests

from auto_ontology.utils.llm_invoke import safe_invoke_text_nr, RETRY_MAX_ATTEMPTS

logger = logging.getLogger(__name__)

_KB_COVERAGE_PROMPT = """\
User question (for context only — use it to understand what each term means in this query):
{question}

External knowledge:
{formatted_kb}

For each term below, identify the external knowledge entries that directly define or provide \
the formula/threshold for that term as it is used in the question above. Only include entries \
whose definition or description gives the exact meaning, calculation, or threshold — not \
entries that merely mention or relate to the concept. Use the entry's description and the \
question's domain context to verify relevance, not just name similarity.

Example: for "item weight", include "Unit Weight Index (UWI)" \
(it defines the formula) but NOT "Shipment Volume Index" (it only relates to conditions).

Terms:
{entity_list}

Output in exactly this format (one line per term, entry names after YES separated by " ;; "):

<term>: YES ;; <entry name 1> ;; <entry name 2> ;; ...
<term>: NO
...\
"""

# Unicode hyphen/dash variants that LLMs commonly emit instead of ASCII '-'
_UNICODE_HYPHENS = str.maketrans(
    {
        "‐": "-",  # hyphen
        "‑": "-",  # non-breaking hyphen
        "‒": "-",  # figure dash
        "–": "-",  # en dash
        "—": "-",  # em dash
        "―": "-",  # horizontal bar
    }
)


def _norm_key(s: str) -> str:
    """Lowercase + collapse unicode hyphens to ASCII for reliable key matching."""
    return unicodedata.normalize("NFC", s).translate(_UNICODE_HYPHENS).lower()


def _parse_kb_entries(formatted_kb: str) -> dict[str, str]:
    """Parse formatted KB string into {normalized_entry_name: full_entry_text}."""
    entries: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in formatted_kb.splitlines():
        if line.startswith("- "):
            if current_name is not None:
                entries[_norm_key(current_name)] = "\n".join(current_lines)
            current_name = line[2:].strip()
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        entries[_norm_key(current_name)] = "\n".join(current_lines)
    return entries


def _slim_kb_for_coverage(formatted_kb: str) -> str:
    """Strip Definition lines so the coverage prompt is ~3x smaller.

    The coverage LLM only needs entry names and descriptions to decide
    which KB entries match an extracted entity. The full definitions are
    looked up separately from the original formatted_kb, so nothing is lost.
    """
    return "\n".join(
        line
        for line in formatted_kb.splitlines()
        if not line.startswith("  Definition:")
    )


_MAX_CHILDREN_PER_PARENT = 5


def _filter_covered_by_external_knowledge(
    entities: list[str],
    formatted_kb: str,
    question: str = "",
    children_map: dict[str, list[str]] | None = None,
) -> tuple[set[str], str, dict[str, list[str]]]:
    """Check which entities are covered by external knowledge.

    Returns (covered_set, relevant_knowledge_text, entry_to_original_terms).
    entry_to_original_terms maps each confirmed KB entry name to the original
    natural-language terms that matched it, so callers can annotate cumulative_grounded_knowledge.
    The LLM outputs YES | <entry name>
    for each covered term; we look up the verbatim entry ourselves so the content
    is never hallucinated. Uses the non-reasoning model for speed.

    When children_map is provided, child entries of any matched parent are appended
    to relevant_knowledge_text with their full text (name + description + definition),
    bypassing the coverage LLM. Capped at _MAX_CHILDREN_PER_PARENT per parent.
    """
    entity_list = "\n".join(f"- {e}" for e in entities)
    slim_kb = _slim_kb_for_coverage(formatted_kb)
    prompt = _KB_COVERAGE_PROMPT.format(
        formatted_kb=slim_kb,
        entity_list=entity_list,
        question=question or "(not provided)",
    )

    def _is_valid_coverage_response(r: str, n_entities: int) -> bool:
        """True when r looks like structured YES/NO lines, not prose."""
        lines = [line for line in r.splitlines() if line.strip()]
        if not lines:
            return False
        # A valid response has ~n_entities lines. Responses with far more lines
        # are chain-of-thought prose that happens to contain ": YES"/": NO" substrings.
        if len(lines) > n_entities * 4 + 10:
            return False
        structured = sum(
            1 for line in lines if ": YES" in line.upper() or ": NO" in line.upper()
        )
        return structured >= max(1, n_entities // 2)

    response = ""
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            response = safe_invoke_text_nr(prompt).strip()
            if response and _is_valid_coverage_response(response, len(entities)):
                break
            if response:
                logger.warning(
                    "Clarify — coverage LLM returned prose on attempt %d/%d — retrying",
                    attempt + 1,
                    RETRY_MAX_ATTEMPTS,
                )
            else:
                logger.warning(
                    "Clarify — coverage LLM returned empty on attempt %d/%d — retrying",
                    attempt + 1,
                    RETRY_MAX_ATTEMPTS,
                )
            response = ""
        except requests.exceptions.ReadTimeout:
            logger.warning(
                "Clarify — coverage LLM timed out on attempt %d/%d",
                attempt + 1,
                RETRY_MAX_ATTEMPTS,
            )
        except Exception as e:
            logger.error(
                "Clarify — coverage LLM error on attempt %d/%d: %s",
                attempt + 1,
                RETRY_MAX_ATTEMPTS,
                e,
            )
            break
        if attempt < RETRY_MAX_ATTEMPTS - 1:
            time.sleep(2 ** (attempt + 1))
        else:
            logger.error(
                "Clarify — coverage LLM failed after %d attempts; treating all entities as unresolvable",
                RETRY_MAX_ATTEMPTS,
            )
    logger.debug("Clarify — coverage LLM raw response:\n%s", response)

    kb_entries = _parse_kb_entries(formatted_kb)

    # Parse term → claimed entry names from coverage response.
    # Do NOT mark a term as covered yet — only do so after the KB lookup confirms
    # the entry actually exists. The coverage LLM sometimes hallucinates entry names
    # from its training data (e.g. SNQI when it is masked from the KB).
    term_to_entry_names: dict[str, list[str]] = {}
    for line in response.splitlines():
        upper = line.upper()
        if ": YES" not in upper:
            continue
        # Strip an optional leading "COVERAGE:" prefix the LLM sometimes emits
        stripped = re.sub(r"^coverage\s*:\s*", "", line, count=1, flags=re.IGNORECASE)
        term = stripped.split(":")[0].strip().lstrip("- ").lower()
        entry_names: list[str] = []
        if ";;" in line:
            after_yes = line.split("YES", 1)[1]
            for entry_name in after_yes.split(";;"):
                name = _norm_key(entry_name.strip())
                if name:
                    entry_names.append(name)
        term_to_entry_names[term] = entry_names

    # Look up verbatim entries; only mark a term covered when a real KB entry is found.
    logger.debug("Clarify — term_to_entry_names: %s", term_to_entry_names)
    covered: set[str] = set()
    relevant_lines: list[str] = []
    seen_names: set[str] = set()
    entry_to_original_terms: dict[str, list[str]] = {}
    # Normalize children_map keys once for efficient lookup
    norm_children_map: dict[str, list[str]] = (
        {_norm_key(k): v for k, v in children_map.items()} if children_map else {}
    )
    for term, entry_names in term_to_entry_names.items():
        for entry_name in entry_names:
            match = next(
                (
                    k
                    for k in kb_entries
                    if k.startswith(entry_name) or entry_name.startswith(k)
                ),
                None,
            )
            logger.debug("Clarify — lookup %r → match=%r", entry_name, match)
            if match:
                # Always record which original term matched this entry, even if the
                # entry is a duplicate (seen_names dedup below).
                entry_to_original_terms.setdefault(match, []).append(term)
            if match:
                if match not in seen_names:
                    seen_names.add(match)
                    relevant_lines.append(kb_entries[match])
                    covered.add(term)  # confirmed: real KB entry exists
                # Inject children even if parent text was already added (dedup via seen_names
                # prevents duplicate text, but grandchildren would be silently skipped if we
                # only entered this block on first sight of the parent).
                children = norm_children_map.get(match, [])[:_MAX_CHILDREN_PER_PARENT]
                if children:
                    logger.debug(
                        "Clarify — KB entry %r has %d child(ren)", match, len(children)
                    )
                for child_text in children:
                    child_name = _norm_key(
                        child_text.split("\n")[0].lstrip("- ").strip()
                    )
                    if child_name and child_name not in seen_names:
                        seen_names.add(child_name)
                        relevant_lines.append(child_text)
                        logger.debug(
                            "Clarify — injected child KB entry: %r (parent: %r)",
                            child_name,
                            match,
                        )
                    # Inject grandchildren — covers 2-level KB hierarchies (e.g. PAR→CGPI→SPR).
                    grandchildren = (
                        norm_children_map.get(child_name, [])[:_MAX_CHILDREN_PER_PARENT]
                        if child_name
                        else []
                    )
                    if grandchildren:
                        logger.debug(
                            "Clarify — KB entry %r has %d grandchild(ren) via %r",
                            child_name,
                            len(grandchildren),
                            match,
                        )
                    for gc_text in grandchildren:
                        gc_name = _norm_key(gc_text.split("\n")[0].lstrip("- ").strip())
                        if gc_name and gc_name not in seen_names:
                            seen_names.add(gc_name)
                            relevant_lines.append(gc_text)
                            logger.debug(
                                "Clarify — injected grandchild KB entry: %r (child: %r, parent: %r)",
                                gc_name,
                                child_name,
                                match,
                            )
                        # Inject great-grandchildren — covers 3-level KB hierarchies.
                        great_grandchildren = (
                            norm_children_map.get(gc_name, [])[
                                :_MAX_CHILDREN_PER_PARENT
                            ]
                            if gc_name
                            else []
                        )
                        if great_grandchildren:
                            logger.debug(
                                "Clarify — KB entry %r has %d great-grandchild(ren) via %r",
                                gc_name,
                                len(great_grandchildren),
                                match,
                            )
                        for ggc_text in great_grandchildren:
                            ggc_name = _norm_key(
                                ggc_text.split("\n")[0].lstrip("- ").strip()
                            )
                            if ggc_name and ggc_name not in seen_names:
                                seen_names.add(ggc_name)
                                relevant_lines.append(ggc_text)
                                logger.debug(
                                    "Clarify — injected great-grandchild KB entry: %r (grandchild: %r, parent: %r)",
                                    ggc_name,
                                    gc_name,
                                    match,
                                )

    relevant_knowledge_text = "\n".join(relevant_lines)
    logger.info("Clarify — external_kb covers: %s", covered or "none")
    logger.debug(
        "Clarify — relevant_knowledge_text stored (%d chars): %r",
        len(relevant_knowledge_text),
        relevant_knowledge_text[:300] if relevant_knowledge_text else "",
    )
    return covered, relevant_knowledge_text, entry_to_original_terms


def _compact_schema(db_schema: str) -> str:
    """Return 'table: col1, col2, ...' lines — enough for the decide-LLM to know
    what columns exist without the verbosity of DDL + sample rows."""
    result: list[str] = []
    current_table: str | None = None
    columns: list[str] = []
    in_ddl = False  # True only between CREATE TABLE ( ... );
    for line in db_schema.splitlines():
        m = re.match(r'CREATE\s+TABLE\s+["\']?(\w+)["\']?\s*\(', line, re.IGNORECASE)
        if m:
            if current_table and columns:
                result.append(f"{current_table}: {', '.join(columns)}")
            current_table = m.group(1)
            columns = []
            in_ddl = True
        elif in_ddl:
            stripped = line.strip()
            # The closing "); " line ends the DDL block
            if stripped.startswith(");"):
                in_ddl = False
                continue
            if not stripped or stripped.startswith(
                ("PRIMARY KEY", "FOREIGN KEY", "CONSTRAINT", ")")
            ):
                continue
            col_name = stripped.split()[0]
            if col_name:
                columns.append(col_name)
    if current_table and columns:
        result.append(f"{current_table}: {', '.join(columns)}")
    return "\n".join(result) or db_schema[:3000]


def expand_kb_with_children(
    formatted_kb: str, children_map: dict[str, list[str]]
) -> str:
    """Return formatted_kb with child entries appended for every parent already present.

    Used by the debug/grounding path where the coverage LLM sees the full KB and
    children need to be visible inline. Deduplicates by normalized entry name.
    """
    if not children_map or not formatted_kb:
        return formatted_kb
    existing = set(_parse_kb_entries(formatted_kb).keys())
    extra: list[str] = []
    for parent_name, child_texts in children_map.items():
        if _norm_key(parent_name) not in existing:
            continue
        for child_text in child_texts[:_MAX_CHILDREN_PER_PARENT]:
            child_name = _norm_key(child_text.split("\n")[0].lstrip("- ").strip())
            if child_name and child_name not in existing:
                existing.add(child_name)
                extra.append(child_text)
    if not extra:
        return formatted_kb
    return formatted_kb + "\n" + "\n".join(extra)
