<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Answering from past conversations in the ReAct agent

A plan, not a built thing. Nothing here is implemented yet.

## What this is for

Today the agent answers from two sources and chooses between them itself: it
writes SQL against the data, or it answers in words from the catalog. A third
source exists in the system and the agent cannot reach it — the semantic graph
built from past conversations, which `auto_ontology.retrieval.conversation`
searches and the BEAM benchmark scores.

The gap shows in questions that name no table. Asked *what did we decide about
the rate limit*, the answer is in a past conversation, the agent has no way to
look there, and it either writes a query against a table that does not hold the
answer or says it cannot find one.

The goal is that the agent decides, per question, whether to search past
conversations, search the catalog, or write SQL — and can do more than one of
them before it answers. Off by default, because most deployments have no
conversation memory and a tool offered without a source behind it is a
distraction the model pays for on every question.

## The shape it takes

This is the `information_answers` design a second time. That is deliberate:
there is already a working answer in the codebase to "let the agent answer in
words from somewhere other than the data", and a second mechanism that does the
same job differently is worse than a second use of the first one.

`information_answers` works like this today:

- a flag on `ToolContext`, set by the caller;
- when on, `retrieve` gains a `purpose` argument and `submit` makes `sql`
  optional, so prose can be the answer;
- the system prompt explains which questions call for which;
- a second flag, `information_retrieved`, records that the right view was
  actually consulted, so an SQL-less answer cannot be invented.

Conversations reuse all four.

### One flag, and no name to plumb

`ToolContext` gains `conversation_answers: bool = False`, next to
`information_answers`. `agent.py` sets it from pipeline state, the same line
and the same way.

A boolean is enough, and this is the part worth being explicit about. The
semantic store resolves a conversation memory through `catalog_database` — the
same table the SQL side resolves `database_name` against:

```python
def _database_id(database_name: str) -> str | None:
    rows = store().query_read(
        select(s.catalog_database.c.id).where(
            s.catalog_database.c.name == database_name
        )
    )
```

So `ctx.database_name` already names the memory to search. No second field, no
env var, nothing for the caller to keep in sync.

This assumes the conversations were ingested under the same catalog database as
the structured data. That is the end state being built towards and not yet true
everywhere — the BEAM runs ingest under their own name, `beam_100K_2`. Until
then the failure is benign rather than loud: `search_semantic` returns empty
hits when a database has no semantic layer instead of raising, so the tool
reports that it found nothing and the model moves on to SQL.

### `purpose` gains a third value

`retrieve` already takes `purpose`, today `sql` or `information`. It gains
`conversation` when the flag is on. The model picks per call and can call
`retrieve` twice with different purposes on one question, which is the whole
point: nobody knows in advance whether *what did we decide about the rate limit*
is a catalog question or a conversation one.

One wrinkle. `RETRIEVE_WITH_PURPOSE_TOOL` is a module-level constant built from
`PURPOSES`. A third value offered only sometimes means building the schema per
episode from the flags rather than once at import. Small, but it is a change in
kind and not a line.

### The new code is one module

`react/conversation.py`, sibling to `react/information.py` and the same shape:
search, then render what was found as itself.

```python
def conversation(ctx: ToolContext, query: str) -> str:
    """What past conversations hold on *query*, as an observation."""
```

It calls `search_semantic(query, database_name=ctx.database_name)` and renders
the note and the turns behind it.

The budget needs care. `max_observation_chars()` is 8,000 and conversation
turns are large — BEAM's own reader is given 29,000 characters of them. So this
needs the two-tier rendering `information.py` already uses for terms, where the
nearest couple of hits come back in full and the rest as one line each
(`_FULL_TERMS = 2`, `_term_block` against `_term_line`). A conversation search
that fills the window leaves no room for the tables a SQL answer still needs.

### Three gates are catalog-shaped and would block it

The answering machinery needs nothing new — `submit` already accepts an empty
`sql` with a prose `response`, and that *is* a conversation answer. But three
places currently say "catalog" where they mean "a source other than the data":

1. `prompts.information_answers_offered()` decides whether prose may be
   submitted at all. It has to be true when conversation search is on, even if
   catalog answers are off.
2. `loop._retrieve` dispatches on `purpose == "information"`. It needs the
   third branch, and it should stay one function rather than becoming a third
   tool (see below).
3. The guard in `submit` refuses an SQL-less answer unless
   `ctx.information_retrieved`:

   ```
   "Not accepted. An answer without SQL has to come from the catalog:
    retrieve what you will say first (purpose=information), or write a query."
   ```

   This needs a `conversation_retrieved` flag set the same way, and the guard
   widened to accept either. Missing this one is the subtle failure: the model
   searches conversations, finds the answer, and is then refused permission to
   say it.

### The prompt

`_INFORMATION_OPENING` frames a two-way choice, catalog or SQL. It needs a
third source and a rule for when a past conversation is the right one — roughly
that a question about what was said, decided, asked for or preferred is a
conversation question, where a question about what the data contains is not.
Added only when the flag is on, so a deployment without conversations reads the
prompt it reads today.

## What to keep straight

**`purpose` is a rendering choice over one dispatch, not three tools.** The
stated direction is that conversations and structured data become one graph.
When that happens a view is cheap to collapse into another view, and three
sibling tools are not. Keeping the branch inside `_retrieve` is what makes the
join cheap later.

**The benchmark number is unaffected, and should stay that way.** The BEAM
harness calls `search_semantic` directly from `run.py` with upstream's reader
prompt, byte-exact, and never goes through the agent. That is what makes 0.856
comparable to cognee's 0.794 against a bar of 0.79. Routing the benchmark
through the agent would be a different arm and a different number; the
one-shot path stays as the headline and anything agentic is measured beside it.

## Order of work

1. `conversation_answers` and `conversation_retrieved` on `ToolContext`, set in
   `agent.py`.
2. `react/conversation.py`, with the two-tier rendering and the budget.
3. The `purpose` enum built per episode; the `_retrieve` branch.
4. The three gates widened.
5. The prompt clause.
6. Tests: the tool is absent when the flag is off; a conversation search sets
   `conversation_retrieved`; a prose answer is refused before any search and
   accepted after one; the observation stays inside the budget.
