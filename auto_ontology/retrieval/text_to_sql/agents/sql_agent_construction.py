# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic replacement for the single-call SQL construction node.

Drop-in for :class:`SQLFromCandidatesAgent`: same prompt, same
:class:`SQLGenerationModel` out, same ``constructable``/``unconstructable``
decision, so validation, execution and the reconstruction loop downstream do
not know which one ran. The difference is that the model may look at the
database before it commits instead of only after a query has already failed.

The reconstruction loop stays in place behind this. The agent should make it
fire less often; it is not meant to replace it, and a question the agent gets
wrong still gets the same repair rounds it gets today.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from auto_ontology.retrieval.text_to_sql.agents.sql_from_semantic import (
    SQLFromCandidatesAgent,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.sql_agent.deep import run_deep_sql_agent
from auto_ontology.retrieval.text_to_sql.sql_agent.ensemble import (
    run_ensemble_sql_agent,
)
from auto_ontology.retrieval.text_to_sql.sql_agent.loop import run_sql_agent
from auto_ontology.retrieval.text_to_sql.sql_agent.tools import SQLAgentToolbox
from auto_ontology.retrieval.text_to_sql.state import AgentState, get_original_question

logger = logging.getLogger(__name__)

# Separate from the module logger so a run's trajectories can be pulled out of
# the log on their own — one JSON object per question, nothing else at this
# name. See ``_log_trajectory``.
trajectory_logger = logging.getLogger(f"{__name__}.trajectory")

# Graph node name this agent is registered under in ``text_to_sql_graph.create_graph``
# (NOT ``self.agent_name``, which is a separate internal/logging name) — must match
# so ``stream_agent_response`` can attribute this agent's recorded thoughts to the
# right step event and ``NODE_LABELS`` entry. It shares the node with the
# single-call agent it replaces, which is what keeps the two swappable.
_GRAPH_NODE_NAME = "construct_sql_from_candidates"


class SQLAgentConstructionAgent(BaseAgent):
    """Construct SQL through a bounded tool-calling loop."""

    def __init__(self) -> None:
        super().__init__("sql_agent_construction")
        # Composition, not inheritance: the only thing wanted from the
        # single-call agent is its prompt and its input contract, and
        # subclassing would also inherit an ``execute`` that must never run.
        self._baseline = SQLFromCandidatesAgent()

    def validate_input(self, state: AgentState) -> bool:
        return self._baseline.validate_input(state)

    def _run(
        self, llm: Any, messages: list, toolbox: SQLAgentToolbox, state: AgentState
    ) -> Any:
        """Drive the agent. Overridden to swap in a different one.

        A seam rather than a second copy of ``execute``: everything around the
        call — the prompt, the fallback to the single call, the trajectory log,
        the path_state contract — is what makes two agents comparable, and is
        the part that must not drift between them.
        """
        return run_sql_agent(llm, messages, toolbox)

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        # Construction is SQL generation, so it takes the reasoning model even
        # though the rest of the graph runs on the cheap one. Writing the query
        # is the step the split deliberately keeps expensive.
        llm = state.get("reasoning_llm") or state["llm"]
        messages = self._baseline.build_messages(state)
        toolbox = SQLAgentToolbox.from_state(state)

        try:
            outcome = self._run(llm, messages, toolbox, state)
        except Exception:  # noqa: BLE001
            # A loop that breaks must not cost the question. The single call
            # is the path every other run already takes, so falling back to it
            # degrades to today's behaviour rather than to no answer.
            self.logger.error(
                "%s failed — falling back to single-call construction",
                self.agent_name,
                exc_info=True,
            )
            return self._baseline.execute(state)
        finally:
            toolbox.close()

        self._log_trajectory(path_state, toolbox, outcome, get_original_question(state))

        self.logger.info(
            "SQL agent finished: stop=%s turns=%d sql_calls=%d probes=%d "
            "tables_discovered=%d",
            outcome.stop_reason,
            outcome.turns,
            toolbox.sql_calls,
            toolbox.executor.calls,
            len(toolbox.discovered_tables),
        )

        if outcome.result is None:
            return {
                "path_state": {
                    **path_state,
                    "unconstructable_explanation": outcome.unconstructable_reason
                    or "The agent could not construct a query.",
                },
                "decision": "unconstructable",
            }

        record_thought(path_state, _GRAPH_NODE_NAME, outcome.result.thought.strip())

        return {
            "messages": messages,
            "path_state": {
                **path_state,
                "sql_generation_result": outcome.result,
                # search_tables may have pulled in tables the retrieval step
                # missed; they have to land here or validation cannot resolve
                # them against the catalog.
                "relevant_tables": toolbox.relevant_tables,
                "custom_analyses_used": [],
            },
            "decision": "constructable",
        }

    def _log_trajectory(
        self,
        path_state: Dict[str, Any],
        toolbox: SQLAgentToolbox,
        outcome: Any,
        question: str = "",
    ) -> None:
        """Emit the whole loop as one JSON line, and keep it on path_state.

        Both, because they answer different questions. The log line is what an
        offline audit reads across a thousand questions; the path_state copy is
        what a single failing question's result row carries with it.

        The question text is recorded alongside ``task_id`` because the harness
        does not set a task id — it logs as ``null`` — and a trajectory that
        cannot be matched back to the row it produced cannot be audited at all.
        """
        trajectory = {
            "task_id": path_state.get("task_id"),
            "question": question,
            "db": path_state.get("target_db"),
            "stop_reason": outcome.stop_reason,
            "turns": outcome.turns,
            "sql_calls": toolbox.sql_calls,
            "probe_calls": toolbox.executor.calls,
            "discovered_tables": [
                t.get("name") for t in toolbox.discovered_tables
            ],
            "steps": outcome.trajectory,
        }
        path_state["sql_agent_trajectory"] = trajectory
        try:
            trajectory_logger.info(json.dumps(trajectory, default=str))
        except Exception:  # noqa: BLE001 — logging must never cost a question
            logger.warning("Failed to serialise SQL agent trajectory", exc_info=True)


class SQLDeepAgentConstructionAgent(SQLAgentConstructionAgent):
    """Construct SQL through the deep agent instead of the bounded loop.

    Same node, same prompt, same contract out — so the two agentic paths and
    the single call are three arms of one experiment rather than three
    pipelines. What differs is only what happens between the prompt and the
    answer; see :mod:`..sql_agent.deep` for why that difference is worth a
    separate arm.
    """

    def __init__(self) -> None:
        super().__init__()
        self.agent_name = "sql_deep_agent_construction"
        self.logger = logging.getLogger(
            f"{__name__}.{self.agent_name}"
        )

    def _run(
        self, llm: Any, messages: list, toolbox: SQLAgentToolbox, state: AgentState
    ) -> Any:
        return run_deep_sql_agent(llm, messages, toolbox, state)


class SQLEnsembleConstructionAgent(SQLAgentConstructionAgent):
    """Construct SQL by writing it several ways and picking by execution.

    The third arm. Where the loop and the deep agent both spend their budget
    improving one query, this one runs the deep agent once and then spends the
    rest rewriting its query several ways from what it found, letting the
    database arbitrate between them; see :mod:`..sql_agent.ensemble` for the
    measurement that says the remaining errors live where independently
    written queries disagree.

    It shares the node, the prompt and the contract with the other two, which
    is the only reason the three can be compared at all. It also shares the
    toolbox, because the exploration it runs is a single deep agent's -- so
    the tool counts logged here mean what they mean on the deep arm, and only
    ``turns`` reads differently, counting candidates rather than model turns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.agent_name = "sql_ensemble_construction"
        self.logger = logging.getLogger(f"{__name__}.{self.agent_name}")

    def _run(
        self, llm: Any, messages: list, toolbox: SQLAgentToolbox, state: AgentState
    ) -> Any:
        return run_ensemble_sql_agent(
            llm, messages, toolbox, state, question=get_original_question(state)
        )


__all__ = [
    "SQLAgentConstructionAgent",
    "SQLDeepAgentConstructionAgent",
    "SQLEnsembleConstructionAgent",
]
