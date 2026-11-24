# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from auto_ontology.retrieval.text_to_sql.prompts import format_dialect_rules
from auto_ontology.retrieval.text_to_sql.react import loop, prompts
from auto_ontology.retrieval.text_to_sql.react import run_sql as run_sql_module
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.mode import EpisodeMode
from auto_ontology.retrieval.text_to_sql.tests.react_fakes import (
    SHOP_CATALOG,
    SHOP_SCRIPT,
    FakeCatalog,
    ScriptedModel,
    SQLiteConnector,
    call,
    show_shop_tables,
)

FINAL_SQL = (
    "SELECT SUM(o.amount - o.discount) FROM orders o "
    "JOIN customers c ON c.customer_id = o.cust_ref WHERE c.status = 'ACTIVE'"
)


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> ToolContext:
    catalog = FakeCatalog(SHOP_CATALOG)
    monkeypatch.setattr(run_sql_module, "catalog_for", lambda *_: catalog)
    monkeypatch.setattr(
        loop,
        "retrieve",
        lambda _ctx, query, kinds=None: f"TABLE orders (for {query})",
    )
    # The real lookup needs a retriever; one that raised would also leave the
    # information flag unset, which the answer-without-SQL tests depend on.
    monkeypatch.setattr(
        loop, "information", lambda _ctx, query, kinds=None: f"TERM {query}"
    )
    return show_shop_tables(ToolContext(connectors=[SQLiteConnector(SHOP_SCRIPT)]))


def _reply(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def test_search_check_and_submit(ctx: ToolContext) -> None:
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="revenue per active customer")),
            _reply(call("run_sql", "2", sql="SELECT DISTINCT status FROM customers")),
            _reply(
                call(
                    "submit",
                    "3",
                    sql=FINAL_SQL,
                    response="Revenue of active customers.",
                )
            ),
        ]
    )

    result = loop.run_episode(model, ctx, "What is revenue from active customers?")

    assert result.reason == "submitted"
    assert result.tool_calls == 3
    assert result.submission.sql == FINAL_SQL
    assert json.loads(result.submission.db_response.result[0]) == [
        {"SUM(o.amount - o.discount)": 160.0}
    ]
    tool_results = [m.content for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert tool_results[0] == "TABLE orders (for revenue per active customer)"
    assert "ACTIVE" in tool_results[1]


def test_cost_status_follows_the_question_and_each_tool_result(
    ctx: ToolContext,
) -> None:
    left = iter([9, 8])
    ctx.cost_status = lambda: f"BUDGET: {next(left)} coins left."
    model = ScriptedModel(
        [
            _reply(call("run_sql", "1", sql="SELECT DISTINCT status FROM customers")),
            _reply(call("submit", "2", sql=FINAL_SQL, response="x")),
        ]
    )

    loop.run_episode(model, ctx, "q")

    assert model.seen[0][1].content.endswith("BUDGET: 9 coins left.")
    tool_result = [m.content for m in model.seen[-1] if isinstance(m, ToolMessage)][0]
    assert "ACTIVE" in tool_result
    assert tool_result.endswith("BUDGET: 8 coins left.")


def test_no_cost_status_leaves_messages_unchanged(ctx: ToolContext) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(model, ctx, "q")

    assert model.seen[0][1].content == prompts.user_message("q", None)


def test_rejected_submission_lets_the_model_fix_it(ctx: ToolContext) -> None:
    model = ScriptedModel(
        [
            _reply(call("submit", "1", sql="SELECT revenue FROM orders", response="x")),
            _reply(call("submit", "2", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    rejection = [t for t in result.trajectory if t["role"] == "tool"][0]["content"]
    assert rejection.startswith("Not accepted.")
    assert "Unknown column: revenue" in rejection


def test_repeated_call_is_not_run_again(ctx: ToolContext) -> None:
    same = dict(query="orders")
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", **same)),
            _reply(call("retrieve", "2", **same)),
            _reply(call("submit", "3", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.tool_calls == 3
    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[0] == "TABLE orders (for orders)"
    assert observations[1].startswith("You already made this exact call")


def test_a_model_that_only_repeats_itself_runs_out_of_calls(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_TOOL_CALLS", "3")
    model = ScriptedModel(
        [_reply(call("retrieve", str(n), query="orders")) for n in range(50)]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "budget_exhausted"
    assert result.tool_calls == 3
    assert len(model.seen) == 4


def test_budget_ends_with_a_submit_only_turn(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_TOOL_CALLS", "2")
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="a")),
            _reply(call("retrieve", "2", query="b")),
            _reply(call("submit", "3", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    assert model.bound_tool_names[-1] == ["submit"]


def test_budget_exhausted_without_submit(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_TOOL_CALLS", "1")
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="a")),
            _reply(call("submit", "2", sql="SELECT nope FROM orders", response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.submission is None
    assert result.reason == "budget_exhausted"


def test_one_answer_attempt_per_reply_after_the_budget(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_TOOL_CALLS", "1")
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="a")),
            _reply(
                call("submit", "2", sql="SELECT nope FROM orders", response="x"),
                call("submit", "3", sql="SELECT nope2 FROM orders", response="x"),
                call("submit", "4", sql=FINAL_SQL, response="x"),
            ),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.submission is None
    assert result.reason == "budget_exhausted"
    assert result.tool_calls == 2
    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[1].startswith("Not accepted.")
    assert (
        observations[2] == observations[3] == prompts.out_of_calls_for(EpisodeMode.SQL)
    )


def test_model_that_never_calls_a_tool_gives_up(ctx: ToolContext) -> None:
    model = ScriptedModel([AIMessage(content="hmm")] * 3)

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "no_submission"
    assert result.tool_calls == 0


def test_unknown_tool_and_bad_arguments_are_observations(ctx: ToolContext) -> None:
    model = ScriptedModel(
        [
            _reply(call("delete_everything", "1")),
            _reply(call("run_sql", "2", query="SELECT 1")),
            _reply(call("submit", "3", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[0].startswith("Unknown tool")
    assert observations[1].startswith("Wrong arguments for run_sql")
    assert result.reason == "submitted"


def test_evidence_reaches_the_model(ctx: ToolContext) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(model, ctx, "q", evidence="active means status = 'ACTIVE'")

    assert "EVIDENCE\nactive means status = 'ACTIVE'" in model.seen[0][1].content


def test_custom_prompts_reach_the_system_prompt(ctx: ToolContext) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(
        model, ctx, "q", custom_prompts="ORGANIZATION INSTRUCTIONS\nUse USD."
    )

    assert model.seen[0][0].content.endswith("\n\nORGANIZATION INSTRUCTIONS\nUse USD.")


_ANCHORS = [
    {
        "phrase": "active",
        "tbl": "customers",
        "col": "status",
        "stored_value": "ACTIVE",
        "n_rows": "2",
    },
    {"phrase": "gold tier", "kind": "absent"},
]

_EXAMPLES = [
    {
        "question": "How many churned users?",
        "evidence": "churned means state = 'C'",
        "sql": "SELECT COUNT(*)\n  FROM users WHERE state = 'C'",
        "db": "telecom",
    },
    {"question": "No SQL here", "sql": "  "},
]


def test_value_anchors_and_sql_examples_follow_the_evidence(
    ctx: ToolContext,
) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(
        model,
        ctx,
        "q",
        evidence="revenue is amount - discount",
        value_anchors=_ANCHORS,
        sql_examples=_EXAMPLES,
    )

    user = model.seen[0][1].content
    evidence_at = user.index("EVIDENCE\nrevenue is amount - discount")
    anchors_at = user.index("VERIFIED DATABASE VALUES")
    examples_at = user.index("REFERENCE QUERY PATTERNS")
    assert evidence_at < anchors_at < examples_at
    assert '- "active": customers."status" = \'ACTIVE\' — 2 rows' in user
    assert 'stored NOWHERE in this database: "gold tier".' in user
    assert "Never filter on a phrase listed as stored nowhere" in user
    assert "1. [source database: telecom]" in user
    assert "   Evidence: churned means state = 'C'" in user
    assert "   SQL: SELECT COUNT(*) FROM users WHERE state = 'C'" in user
    assert "No SQL here" not in user


def test_without_anchors_or_examples_the_user_message_is_unchanged(
    ctx: ToolContext,
) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(model, ctx, "q", value_anchors=[], sql_examples=[])

    assert model.seen[0][1].content == prompts.user_message("q", None)
    assert model.seen[0][1].content == "QUESTION\nq\n"


def test_row_counts_are_dropped_between_columns_of_one_table() -> None:
    section = prompts.value_anchors_section(
        [
            {
                "phrase": "San Jose",
                "tbl": "schools",
                "col": "City",
                "stored_value": "San Jose",
                "n_rows": "5",
            },
            {
                "phrase": "San Jose",
                "tbl": "schools",
                "col": "MailCity",
                "stored_value": "San Jose",
                "n_rows": "9",
                "description": "mailing city",
                "kind": "contains",
            },
        ]
    )

    assert '- "San Jose": schools."City" = \'San Jose\'\n' in section
    assert (
        '- "San Jose": schools."MailCity" (mailing city) contains it inside '
        "values such as 'San Jose'"
    ) in section
    assert "rows" not in section.split("Use these")[0]
    assert "stored nowhere" not in section


def test_short_answer_rules_reach_the_system_prompt(ctx: ToolContext) -> None:
    model = ScriptedModel(
        [
            _reply(call("submit", "1", sql=FINAL_SQL, response="x")),
            _reply(call("submit", "1", sql=FINAL_SQL, response="x")),
        ]
    )

    loop.run_episode(model, ctx, "q", shorten_answer=True)
    loop.run_episode(model, ctx, "q")

    assert model.seen[0][0].content.endswith(f"\n\n{prompts.SHORT_ANSWER_RULES}")
    assert "OUTPUT SHAPE" not in model.seen[1][0].content


def test_a_prediction_episode_ignores_sql_shaping_inputs(ctx: ToolContext) -> None:
    model = ScriptedModel([AIMessage(content="no tools")] * 3)

    loop.run_episode(
        model,
        ctx,
        "q",
        mode=EpisodeMode.PREDICT,
        value_anchors=_ANCHORS,
        sql_examples=_EXAMPLES,
        shorten_answer=True,
    )

    assert "OUTPUT SHAPE" not in model.seen[0][0].content
    assert model.seen[0][1].content == "QUESTION\nq\n"


@pytest.mark.parametrize("dialect", ["mysql", "sqlite", "oracle"])
def test_dialect_rules_reach_the_model(ctx: ToolContext, dialect: str) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(model, ctx, "q", dialect=dialect)

    system = model.seen[0][0].content
    rules = format_dialect_rules(dialect).strip()
    base = prompts.system_prompt_for(EpisodeMode.SQL, dialect)
    # A dialect with rules gets them after the base prompt; one without gets
    # the base prompt alone.
    assert system == (f"{base}\n\n{rules}" if rules else base)
    assert ("Do NOT CAST a computed value" in system) == (dialect == "mysql")


@pytest.fixture
def no_submit_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_SUBMIT_TOOL", "false")


def test_without_submit_a_final_run_is_the_answer(
    ctx: ToolContext, no_submit_tool: None
) -> None:
    model = ScriptedModel(
        [
            _reply(call("run_sql", "1", sql="SELECT DISTINCT status FROM customers")),
            _reply(call("run_sql", "2", sql=FINAL_SQL, final=True, response="r")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    assert result.submission.sql == FINAL_SQL
    assert result.submission.response == "r"
    assert model.bound_tool_names[0] == ["retrieve", "run_sql"]
    assert "final=true" in model.seen[0][0].content
    assert "submit:" not in model.seen[0][0].content


def test_without_submit_a_failed_final_run_lets_the_model_fix_it(
    ctx: ToolContext, no_submit_tool: None
) -> None:
    model = ScriptedModel(
        [
            _reply(call("run_sql", "1", sql="SELECT revenue FROM orders", final=True)),
            _reply(call("run_sql", "2", sql=FINAL_SQL, final="true")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    rejection = [t for t in result.trajectory if t["role"] == "tool"][0]["content"]
    assert rejection.startswith("Not accepted.")


def test_without_submit_the_sql_run_cap_does_not_block_the_answer(
    ctx: ToolContext, no_submit_tool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_SQL_RUNS", "1")
    model = ScriptedModel(
        [
            _reply(call("run_sql", "1", sql="SELECT 1")),
            _reply(call("run_sql", "2", sql="SELECT 2")),
            _reply(call("run_sql", "3", sql=FINAL_SQL, final=True)),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert "final=true" in observations[1]
    assert result.reason == "submitted"
    assert result.submission.sql == FINAL_SQL


def test_without_submit_the_last_turn_offers_only_run_sql(
    ctx: ToolContext, no_submit_tool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_TOOL_CALLS", "1")
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="a")),
            _reply(call("run_sql", "2", sql=FINAL_SQL, final=True)),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    assert model.bound_tool_names[-1] == ["run_sql"]


def test_without_submit_auto_mode_answers_with_run_sql_or_predict(
    monkeypatch: pytest.MonkeyPatch, no_submit_tool: None
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_GREP", "true")
    names = lambda **kw: [  # noqa: E731
        t["function"]["name"] for t in prompts.tools_for(EpisodeMode.AUTO, **kw)
    ]

    assert names() == ["retrieve", "run_sql", "grep", "predict"]
    assert names(last_turn=True) == ["run_sql", "predict"]
    prompt = prompts.system_prompt_for(EpisodeMode.AUTO, None)
    assert "- grep:" in prompt and "submit" not in prompt


def test_by_default_submit_is_offered_and_grep_is_not() -> None:
    assert [t["function"]["name"] for t in prompts.tools_for(EpisodeMode.SQL)] == [
        "retrieve",
        "run_sql",
        "submit",
    ]
    prompt = prompts.system_prompt_for(EpisodeMode.SQL, None)
    assert prompt.startswith(prompts._SQL_OPENING.format(dialect="SQL"))
    assert "sql empty" not in prompt


def _information_ctx(ctx: ToolContext) -> ToolContext:
    ctx.information_answers = True
    return ctx


def test_an_information_question_is_answered_in_words(ctx: ToolContext) -> None:
    answer = "Table orders stores each order; amount is its total before discount."
    model = ScriptedModel(
        [
            _reply(
                call(
                    "retrieve",
                    "1",
                    query="what does the orders table hold",
                    purpose="information",
                )
            ),
            _reply(call("submit", "2", sql="", response=answer)),
        ]
    )

    result = loop.run_episode(model, _information_ctx(ctx), "What is in orders?")

    assert result.reason == "submitted"
    assert result.submission.sql == ""
    assert result.submission.response == answer
    assert result.submission.db_response is None
    submit = prompts.tools_for(EpisodeMode.SQL, information_answers=True)[-1][
        "function"
    ]
    assert submit["parameters"]["required"] == ["response"]
    assert "sql empty and the answer in response" in model.seen[0][0].content


def test_information_answers_still_need_a_response(ctx: ToolContext) -> None:
    observation, submission = loop.submit(_information_ctx(ctx), "", response="  ")

    assert submission is None
    assert observation == "Error: sql and response are both empty."


def test_a_query_is_still_run_when_information_answers_are_allowed(
    ctx: ToolContext,
) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="r"))])

    result = loop.run_episode(model, _information_ctx(ctx), "q")

    assert result.submission.sql == FINAL_SQL
    assert result.submission.db_response is not None


def test_an_information_answer_in_prose_is_nudged_to_submit(ctx: ToolContext) -> None:
    answer = "Table orders stores each order."
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="orders", purpose="information")),
            AIMessage(content=answer),
            _reply(call("submit", "2", sql="", response=answer)),
        ]
    )

    result = loop.run_episode(model, _information_ctx(ctx), "What is in orders?")

    assert result.submission.response == answer
    human = [t["content"] for t in result.trajectory if t["role"] == "human"]
    assert human[-1] == prompts.INFORMATION_NUDGE


def test_an_answer_without_sql_needs_a_retrieve_first(ctx: ToolContext) -> None:
    answer = "Table orders stores each order."
    model = ScriptedModel(
        [
            _reply(call("submit", "1", sql="", response=answer)),
            _reply(call("retrieve", "2", query="orders", purpose="information")),
            _reply(call("submit", "3", sql="", response=answer)),
        ]
    )

    result = loop.run_episode(model, _information_ctx(ctx), "What is in orders?")

    assert result.submission.response == answer
    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[0].startswith(
        "Not accepted. An answer without SQL has to come from the catalog"
    )


def test_a_sql_search_does_not_unlock_an_answer_without_sql(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop, "retrieve", lambda _ctx, query, kinds=None: "TABLES")
    monkeypatch.setattr(
        loop, "information", lambda _ctx, query, kinds=None: f"TERM {query}"
    )
    answer = "Table orders stores each order."
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="orders")),
            _reply(call("submit", "2", response=answer)),
            _reply(call("retrieve", "3", query="orders", purpose="information")),
            _reply(call("submit", "4", response=answer)),
        ]
    )

    result = loop.run_episode(model, _information_ctx(ctx), "What is in orders?")

    assert result.submission.response == answer
    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[1].startswith(
        "Not accepted. An answer without SQL has to come from the catalog"
    )


def test_a_failed_information_lookup_does_not_unlock_an_answer_without_sql(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookups = iter([RuntimeError("catalog unavailable"), "TERM order"])

    def information(
        _ctx: ToolContext, query: str, kinds: list[str] | None = None
    ) -> str:
        outcome = next(lookups)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(loop, "information", information)
    answer = "Table orders stores each order."
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="orders", purpose="information")),
            _reply(call("submit", "2", sql="", response=answer)),
            _reply(call("retrieve", "3", query="order", purpose="information")),
            _reply(call("submit", "4", sql="", response=answer)),
        ]
    )

    result = loop.run_episode(model, _information_ctx(ctx), "What is in orders?")

    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[0].startswith("retrieve failed: catalog unavailable")
    assert observations[1].startswith(
        "Not accepted. An answer without SQL has to come from the catalog"
    )
    assert observations[2] == "TERM order"
    assert result.submission.response == answer


def test_an_information_lookup_that_finds_nothing_still_counts(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        loop,
        "information",
        lambda _ctx, query, kinds=None: f"No matches for {query!r}.",
    )
    answer = "The catalog has nothing about shipments."
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="shipments", purpose="information")),
            _reply(call("submit", "2", sql="", response=answer)),
        ]
    )

    result = loop.run_episode(model, _information_ctx(ctx), "Are there shipments?")

    assert result.submission.response == answer


def test_an_answer_can_leave_sql_out(ctx: ToolContext) -> None:
    answer = "Table orders stores each order."
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="orders", purpose="information")),
            _reply(call("submit", "2", response=answer)),
        ]
    )

    result = loop.run_episode(model, _information_ctx(ctx), "What is in orders?")

    assert result.reason == "submitted"
    assert result.submission.sql == ""
    assert result.submission.response == answer


def test_without_information_answers_a_submit_without_sql_is_an_error(
    ctx: ToolContext,
) -> None:
    text, submission = loop.submit(ctx, response="Orders are stored.")

    assert text == "Error: sql is empty."
    assert submission is None


def test_retrieve_for_information_shows_the_information_view(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        loop, "information", lambda _ctx, query, kinds=None: f"TERM {query}"
    )
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="Customer", purpose="information")),
            _reply(call("submit", "2", sql="", response="A customer buys.")),
        ]
    )

    result = loop.run_episode(model, _information_ctx(ctx), "What is a customer?")

    assert result.submission.response == "A customer buys."
    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[0] == "TERM Customer"
    retrieve = prompts.tools_for(EpisodeMode.SQL, information_answers=True)[0]
    purpose = retrieve["function"]["parameters"]["properties"]["purpose"]
    assert purpose["enum"] == ["sql", "information"]
    assert "purpose=information" in model.seen[0][0].content


def test_calculation_only_retrieve_has_no_purpose_and_refuses_one(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop, "information", lambda *_a, **_k: "never")
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="orders", purpose="information")),
            _reply(call("submit", "2", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[0] == (
        "Error: purpose 'information' is not available here. Use sql instead."
    )
    retrieve = prompts.tools_for(EpisodeMode.SQL)[0]
    assert "purpose" not in retrieve["function"]["parameters"]["properties"]


def test_calculation_only_keeps_the_original_prompt() -> None:
    assert prompts.system_prompt_for(EpisodeMode.SQL, "mysql").startswith(
        "You answer a question about a database by writing one mysql SQL query.\n\n"
        "You start with no schema."
    )


def test_calculation_only_keeps_the_query_messages() -> None:
    assert prompts.nudge_for(EpisodeMode.SQL) == prompts.NUDGE
    assert prompts.last_call_for(EpisodeMode.SQL) == prompts.LAST_CALL
    assert (
        prompts.last_call_for(EpisodeMode.SQL, information_answers=True)
        == prompts.INFORMATION_LAST_CALL
    )


def test_calculation_only_rejects_an_answer_without_sql(ctx: ToolContext) -> None:
    observation, submission = loop.submit(ctx, "", response="Orders holds orders.")

    assert submission is None
    assert observation == "Error: sql is empty."
    submit = prompts.tools_for(EpisodeMode.SQL)[-1]["function"]
    assert submit["parameters"]["required"] == ["sql", "response"]
    assert "sql empty" not in prompts.system_prompt_for(EpisodeMode.SQL, None)


def test_information_answers_need_the_submit_tool(
    ctx: ToolContext, no_submit_tool: None
) -> None:
    observation, submission = loop.submit(_information_ctx(ctx), "", response="x")

    assert submission is None
    assert observation == "Error: sql is empty."
    prompt = prompts.system_prompt_for(EpisodeMode.SQL, None, information_answers=True)
    assert "sql empty" not in prompt


@pytest.mark.parametrize("dialect", [None, "postgresql", "mysql", "sqlite"])
def test_column_order_rule_reaches_the_model(
    ctx: ToolContext, dialect: str | None
) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(model, ctx, "q", dialect=dialect)

    assert "same order the question" in model.seen[0][0].content


@pytest.mark.parametrize("dialect", [None, "postgresql", "mysql", "sqlite"])
def test_requested_shape_rule_reaches_the_model(
    ctx: ToolContext, dialect: str | None
) -> None:
    model = ScriptedModel([_reply(call("submit", "1", sql=FINAL_SQL, response="x"))])

    loop.run_episode(model, ctx, "q", dialect=dialect)

    system = model.seen[0][0].content
    assert "Aggregate, GROUP BY or LIMIT only when the question" in system
    assert "do not add a key or id to it" in system


def _one_query_budget(ctx: ToolContext) -> None:
    ctx.can_run_queries = lambda: ctx.sql_runs < 1


def test_out_of_budget_offers_only_submit_and_accepts_without_running(
    ctx: ToolContext,
) -> None:
    _one_query_budget(ctx)
    model = ScriptedModel(
        [
            _reply(call("run_sql", "1", sql="SELECT DISTINCT status FROM customers")),
            _reply(call("submit", "2", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    assert result.submission.sql == FINAL_SQL
    assert result.submission.db_response.result == []
    assert model.bound_tool_names[1] == ["submit"]
    assert model.seen[1][-1].content == prompts.NO_QUERIES_LEFT


def test_out_of_budget_submission_still_has_to_pass_the_schema_checks(
    ctx: ToolContext,
) -> None:
    ctx.can_run_queries = lambda: False
    model = ScriptedModel(
        [
            _reply(call("submit", "1", sql="SELECT revenue FROM orders", response="x")),
            _reply(call("submit", "2", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.submission.sql == FINAL_SQL
    rejection = [t for t in result.trajectory if t["role"] == "tool"][0]["content"]
    assert "Unknown column: revenue" in rejection


def test_out_of_budget_episode_without_a_submission_submits_the_latest_draft(
    ctx: ToolContext,
) -> None:
    _one_query_budget(ctx)
    model = ScriptedModel(
        [_reply(call("run_sql", "1", sql=FINAL_SQL)), AIMessage(content="Stuck.")]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted_draft"
    assert result.submission.sql == FINAL_SQL


def test_free_queries_keep_an_unsubmitted_episode_unanswered(ctx: ToolContext) -> None:
    model = ScriptedModel(
        [_reply(call("run_sql", "1", sql=FINAL_SQL)), AIMessage(content="Stuck.")]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "no_submission"
    assert result.submission is None


class InternalServerError(Exception):
    """Named like the client's error for a throttled gateway, which the retry matches by type."""


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waited: list[float] = []
    monkeypatch.setattr("auto_ontology.utils.llm_invoke.time.sleep", waited.append)
    return waited


def test_a_throttled_model_call_is_retried_and_the_episode_goes_on(
    ctx: ToolContext, sleeps: list[float]
) -> None:
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="revenue")),
            InternalServerError("Error code: 500 - throttling_error 429"),
            InternalServerError("Error code: 500 - throttling_error 429"),
            _reply(call("submit", "2", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "submitted"
    assert len(sleeps) == 2
    # The retried turn saw the earlier search result: nothing was restarted.
    assert any(isinstance(m, ToolMessage) for m in model.seen[-1])


def test_a_model_error_that_is_not_transient_is_not_retried(
    ctx: ToolContext, sleeps: list[float]
) -> None:
    model = ScriptedModel([ValueError("bad request")])

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "model_error"
    assert sleeps == []


def test_retries_end_and_the_episode_gives_up(
    ctx: ToolContext, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MODEL_RETRIES", "3")
    model = ScriptedModel([InternalServerError("throttled")] * 5)

    result = loop.run_episode(model, ctx, "q")

    assert result.reason == "model_error"
    assert len(model.seen) == 3
    assert len(sleeps) == 2


@pytest.fixture
def conversation_ctx(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> ToolContext:
    monkeypatch.setattr(
        loop, "conversation", lambda _ctx, query: f"TURN 14 (for {query})"
    )
    ctx.conversation_answers = True
    return ctx


def test_the_conversation_purpose_appears_only_when_it_is_switched_on() -> None:
    def purpose(**flags: bool) -> list[str] | None:
        retrieve = prompts.tools_for(EpisodeMode.SQL, **flags)[0]
        found = retrieve["function"]["parameters"]["properties"].get("purpose")
        return found["enum"] if found else None

    assert purpose() is None
    assert purpose(information_answers=True) == ["sql", "information"]
    assert purpose(conversation_answers=True) == ["sql", "conversation"]
    assert purpose(information_answers=True, conversation_answers=True) == [
        "sql",
        "information",
        "conversation",
    ]


def test_a_conversation_search_reaches_the_conversation_view(
    conversation_ctx: ToolContext,
) -> None:
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="rate limit", purpose="conversation")),
            _reply(call("submit", "2", response="The cap was raised to 4,800.")),
        ]
    )

    result = loop.run_episode(model, conversation_ctx, "q")

    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[0] == "TURN 14 (for rate limit)"
    assert conversation_ctx.conversation_retrieved is True


def test_the_purpose_is_refused_when_conversations_are_off(ctx: ToolContext) -> None:
    # Tables in place of the conversations asked for would read as nothing
    # having been said.
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="rate limit", purpose="conversation")),
            _reply(call("submit", "2", sql=FINAL_SQL, response="x")),
        ]
    )

    result = loop.run_episode(model, ctx, "q")

    observations = [t["content"] for t in result.trajectory if t["role"] == "tool"]
    assert observations[0] == (
        "Error: purpose 'conversation' is not available here. Use sql instead."
    )
    assert ctx.conversation_retrieved is False


def test_a_conversation_answer_needs_a_conversation_search_first(
    conversation_ctx: ToolContext,
) -> None:
    # Otherwise the model searches, finds the answer, and is refused permission
    # to say it -- or worse, says it without having searched at all.
    refused, submission = loop.submit(conversation_ctx, "", response="4,800.")
    assert submission is None
    assert "past conversations (purpose=conversation)" in refused

    loop._retrieve(conversation_ctx, "rate limit", None, "conversation")
    observation, submission = loop.submit(conversation_ctx, "", response="4,800.")

    assert observation == "Accepted."
    assert submission is not None and submission.sql == ""


def test_the_prompt_names_conversations_only_when_they_are_on() -> None:
    off = prompts.system_prompt_for(EpisodeMode.SQL, "mysql", information_answers=True)
    on = prompts.system_prompt_for(
        EpisodeMode.SQL, "mysql", information_answers=True, conversation_answers=True
    )

    assert "purpose=conversation" not in off
    assert "purpose=conversation" in on
    # The catalog wording an existing deployment was measured on is unchanged.
    assert on.startswith(off[: off.index("\n\nYou start with no schema.")])


def test_conversations_alone_still_allow_an_answer_in_words() -> None:
    prompt = prompts.system_prompt_for(
        EpisodeMode.SQL, "mysql", conversation_answers=True
    )

    assert "sql empty" in prompt
    # The catalog view is not described, because it is not offered.
    assert 'kinds=["term"]' not in prompt


def test_no_calculation_offers_no_way_to_run_a_query() -> None:
    # A tool that can only fail is worse than an absent one: the model spends
    # turns discovering that.
    tools = prompts.tools_for(
        EpisodeMode.SQL, calculation_answers=False, conversation_answers=True
    )

    assert [t["function"]["name"] for t in tools] == ["retrieve", "submit"]
    retrieve = tools[0]["function"]["parameters"]["properties"]
    assert retrieve["purpose"]["enum"] == ["conversation"]
    assert prompts.tools_for(
        EpisodeMode.SQL,
        last_turn=True,
        calculation_answers=False,
        conversation_answers=True,
    ) == [tools[-1]]


def test_no_calculation_gets_a_prompt_that_is_not_about_queries() -> None:
    prompt = prompts.system_prompt_for(
        EpisodeMode.SQL, "mysql", calculation_answers=False, conversation_answers=True
    )

    assert "GROUP BY" not in prompt
    assert "purpose=conversation" in prompt
    # The catalog view is not described, because it is not offered.
    assert "purpose=information" not in prompt
    assert (
        prompts.nudge_for(EpisodeMode.SQL, calculation_answers=False)
        == prompts.NO_SQL_NUDGE
    )


def test_a_conversation_only_episode_answers_without_a_warehouse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The state a database holding transcripts and no tables is in. Nothing
    # here has a connector, and the episode still reaches an answer.
    monkeypatch.setattr(
        loop, "conversation", lambda _ctx, query: f"TURN 14 (for {query})"
    )
    ctx = ToolContext(
        connectors=[],
        database_name="beam",
        calculation_answers=False,
        conversation_answers=True,
    )
    model = ScriptedModel(
        [
            _reply(call("retrieve", "1", query="rate limit", purpose="conversation")),
            _reply(call("submit", "2", response="The cap was raised to 4,800.")),
        ]
    )

    result = loop.run_episode(model, ctx, "what cap did we agree?")

    assert result.reason == "submitted"
    assert result.submission is not None
    assert result.submission.sql == ""
    assert result.submission.response == "The cap was raised to 4,800."
