# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SQL Unconstructable Response Agent

This agent generates a response when SQL cannot be constructed from available data.
"""

import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage

from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.state import AgentState

logger = logging.getLogger(__name__)


class SQLUnconstructableAgent(BaseAgent):
    """
    Agent that generates a response when SQL construction fails.

    This agent returns a message explaining that SQL cannot be constructed
    from the available data, optionally including a detailed explanation.

    Input Requirements:
    - path_state["unconstructable_explanation"]: Optional explanation text

    Output:
    - path_state["final_response"]: complete response dict
    - messages: appended AIMessage with the explanation
    """

    def __init__(self):
        super().__init__("sql_unconstructable")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate unconstructable SQL response.

        Returns a message explaining that SQL cannot be constructed,
        using the explanation from path_state if available.

        The explanation goes into ``path_state["final_response"]``, the same
        channel :class:`ResponseAgent` uses, because that is what
        ``_extract_answer`` reads. An empty response reaches the user as a
        generic "something went wrong", and the server declines to persist the
        turn at all, so this node has to fill that field to say anything.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains the final response dict
            - messages: Unconstructable response message
        """
        path_state = state.get("path_state", {})
        unconstructable = path_state.get("unconstructable_explanation", "")

        response_text = (
            unconstructable
            if unconstructable
            else "SQL can't be constructed from the data."
        )

        # Same keys as a successful answer so consumers (persistence, the chart
        # step, the client) can read it without special-casing this path.
        response = {
            "response": response_text,
            "sql_code": "",
            "sql_columns": [],
            "custom_analyses_used": [],
            "sql_response_from_db": None,
        }

        self.logger.info(
            f"Generated unconstructable SQL response: {response_text[:50]}..."
        )

        return {
            "messages": list(state.get("messages") or [])
            + [AIMessage(content=response_text)],
            "path_state": {
                **path_state,
                "final_response": response,
            },
        }
