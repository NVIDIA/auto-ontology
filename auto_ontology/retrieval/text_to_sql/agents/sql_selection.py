# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elect one SQL candidate from the generated pool.

The generator produces N candidates by different strategies; without this node
slot 0 ships and the rest are discarded. Selection runs each candidate and
picks by agreement between their *results*, because candidates that disagree
about the SQL but agree about the answer are evidence for that answer.

Three rules, in order of how much they are worth:

1. Drop candidates that fail to execute, and candidates that come back empty.
   Empty is the expensive case: an over-constrained predicate returns no rows,
   several candidates make that same mistake, and their agreement outvotes the
   one candidate that found the data.
2. Group survivors by result and take the largest group. Different SQL
   returning the same rows is the same answer.
3. Break near-ties on structural fit against the reference examples, whose
   SQL was chosen for this question before any candidate existed. Every other
   signal here is candidates agreeing with each other; this one is not, which
   is why it can move a decision that voting alone gets wrong.

Measured on BIRD dev (1517 questions, 6 candidates): 73.96% shipping slot 0
versus 75.54% with this node, +35 questions fixed against 11 broken,
p=0.0005. No LLM calls -- the cost is one execution per candidate.

What it cannot do: when a single candidate is right and five are wrong, no
vote elects it. That bounds this approach well below the pool's ceiling.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Dict

from auto_ontology.retrieval.text_to_sql import candidate_flags as flags
from auto_ontology.retrieval.text_to_sql.agents.sql_execution import _run_sql
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.sql_skeleton import (
    skeleton,
    skeleton_similarity,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState

_FAILED = "__failed__"
_EMPTY = "__empty__"


def _result_signature(response: Any) -> tuple[str, int]:
    """A hash identifying *response*'s rows, plus the row count.

    Rows are deduplicated and sorted so that two candidates differing only in
    row order or in repeated rows land in the same group -- the same
    equivalence execution accuracy uses.
    """
    if response is None or getattr(response, "error", None):
        return _FAILED, -1
    payload = getattr(response, "result", None) or []
    rows: list = []
    for chunk in payload:
        try:
            parsed = json.loads(chunk)
        except (TypeError, ValueError):
            # A payload that will not parse is still a distinct answer; key it
            # by its text rather than conflating it with an empty result.
            rows.append(str(chunk))
            continue
        if isinstance(parsed, list):
            rows.extend(parsed)
        else:
            rows.append(parsed)
    if not rows:
        return _EMPTY, 0
    canonical = sorted({json.dumps(row, sort_keys=True, default=str) for row in rows})
    digest = hashlib.sha1("\x1f".join(canonical).encode("utf-8")).hexdigest()[:16]
    return digest, len(rows)


def _pattern_fit(sql: str, example_skeletons: list[str]) -> float:
    """How well *sql*'s structure matches the closest reference example."""
    if not example_skeletons:
        return 0.0
    own = skeleton(sql)
    return max(
        (skeleton_similarity(own, other) for other in example_skeletons),
        default=0.0,
    )


class SQLSelectionAgent(BaseAgent):
    """Choose which generated candidate to send on for validation.

    Input:
    - ``path_state["sql_candidates"]``: the generated pool
    - ``path_state["sql_generation_result"]``: slot 0, the incumbent
    - ``connectors``: used to execute candidates

    Output:
    - ``path_state["sql_generation_result"]``: the elected candidate
    - ``path_state["sql_selection"]``: what was decided and why
    """

    def __init__(self):
        super().__init__("sql_selection")

    def validate_input(self, state: AgentState) -> bool:
        # A pool of one, or none, leaves nothing to decide. Not an error: the
        # node sits on the main path and every single-candidate run reaches it.
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state", {}) or {})
        candidates = list(path_state.get("sql_candidates") or [])
        incumbent = path_state.get("sql_generation_result")

        if len(candidates) < 2:
            return {"path_state": path_state, "decision": "constructable"}

        connector = resolve_connector_from_tables(
            path_state.get("relevant_tables", []), state.get("connectors") or []
        )
        example_skeletons = [
            skeleton(example.get("sql") or "")
            for example in (state.get("sql_examples") or [])
            if (example.get("sql") or "").strip()
        ]
        alpha = flags.selection_pattern_alpha()
        drop_empty = flags.selection_drop_empty()

        # Execute once per candidate and record what came back.
        executed: list[dict] = []
        for index, candidate in enumerate(candidates):
            sql = (getattr(candidate, "sql_code", "") or "").strip()
            if not sql:
                continue
            signature, n_rows = _result_signature(_run_sql(sql, connector))
            executed.append(
                {
                    "slot": index,
                    "candidate": candidate,
                    "sig": signature,
                    "rows": n_rows,
                    "fit": _pattern_fit(sql, example_skeletons),
                }
            )

        # Narrow in the order that preserves the most information: prefer
        # candidates that returned rows, fall back to any that executed, and
        # only then to the incumbent.
        usable = [row for row in executed if row["sig"] not in (_FAILED, _EMPTY)]
        if drop_empty:
            pool = usable
        else:
            pool = [row for row in executed if row["sig"] != _FAILED]
        if not pool:
            pool = [row for row in executed if row["sig"] != _FAILED]
        if not pool:
            self.logger.info(
                "No candidate executed; keeping slot 0 of %d.", len(candidates)
            )
            path_state["sql_selection"] = {
                "winner": 0,
                "reason": "no_candidate_executed",
                "n_candidates": len(candidates),
            }
            return {"path_state": path_state, "decision": "constructable"}

        votes: dict[str, float] = defaultdict(float)
        best_fit: dict[str, float] = defaultdict(float)
        for row in pool:
            votes[row["sig"]] += 1.0
            best_fit[row["sig"]] = max(best_fit[row["sig"]], row["fit"])
        scored = {sig: votes[sig] + alpha * best_fit[sig] for sig in votes}
        winning_sig = max(scored.items(), key=lambda item: item[1])[0]
        # Lowest slot within the winning group: slot 0 is the strongest single
        # strategy, so it breaks an intra-group tie in its own favour.
        winner = min(
            (row for row in pool if row["sig"] == winning_sig),
            key=lambda row: row["slot"],
        )

        path_state["sql_selection"] = {
            "winner": winner["slot"],
            "n_candidates": len(candidates),
            "n_executed": len(executed),
            "n_usable": len(usable),
            "n_groups": len(votes),
            "votes": int(votes[winning_sig]),
            "pattern_fit": round(best_fit[winning_sig], 4),
            "reason": "result_vote",
        }
        if winner["candidate"] is not incumbent:
            path_state["sql_generation_result"] = winner["candidate"]
            self.logger.info(
                "Selected candidate %d over slot 0: %d/%d votes across %d "
                "distinct results, pattern fit %.2f.",
                winner["slot"],
                int(votes[winning_sig]),
                len(pool),
                len(votes),
                best_fit[winning_sig],
            )
        else:
            self.logger.info(
                "Kept slot 0: %d/%d votes across %d distinct results.",
                int(votes[winning_sig]),
                len(pool),
                len(votes),
            )
        return {"path_state": path_state, "decision": "constructable"}
