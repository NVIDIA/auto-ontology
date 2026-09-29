<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# auto-ontology-mcp

An [MCP](https://modelcontextprotocol.io) server for
[Auto Ontology](https://github.com/NVIDIA/auto-ontology). It lets any
MCP-capable agent — Cursor, Claude Desktop, an internal agent — ask questions in
natural language about the data a Auto Ontology deployment is connected to, and inspect the
semantic layer behind the answers.

## Install and run

`uvx` fetches, builds, and runs it straight from the repository, so there is
nothing to clone:

```sh
uvx --from "git+https://github.com/NVIDIA/auto-ontology.git#subdirectory=mcp" auto-ontology-mcp
```

From a checkout of this directory, `uvx --from . auto-ontology-mcp` does the same.

> [!NOTE]
> Auto Ontology is NVIDIA-internal today, so this install needs GitHub credentials with
> access to the repository. It becomes a plain `uvx auto-ontology-mcp` once the package is
> published to PyPI.

Run one server; people log in through it. The URL of your Auto Ontology deployment is the
only thing it needs to be told:

```sh
AUTO_ONTOLOGY_API_URL=https://auto_ontology.example.com auto-ontology-mcp
```

## Connect a client

In Cursor (`~/.cursor/mcp.json`) or Claude Desktop
(`claude_desktop_config.json`), point at the URL. There are no credentials in the
config:

```json
{
  "mcpServers": {
    "auto-ontology": {
      "url": "https://auto-ontology-mcp.example/mcp"
    }
  }
}
```

The client offers to sign in, the user gets Auto Ontology's normal login page, and every
call afterwards runs as that person.

Then ask something like *"what does Auto Ontology mean by an active customer, and how many
were there last quarter?"*

## Tools

`ask_question` is the one that answers questions: it runs Auto Ontology's text-to-SQL agent
and returns the answer, the SQL it ran, and the rows. The rest — `search_terms`,
`describe_table`, `check_answerable` and friends — let an agent learn the
vocabulary and check its assumptions first.

Every tool reads, and every tool works through the semantic layer: there is none
for browsing databases, schemas, or raw columns. Nothing here modifies the
glossary, the catalog, or the underlying databases.

## Notes

This server is a plain HTTP client of the Auto Ontology API, so it needs no database
credentials and can run anywhere that can reach your deployment. It holds no
credentials of its own either — Auto Ontology signs each caller in — so it never has more
access than the person calling it.

Full documentation — all configuration variables, the complete tool list, and how
to extend it — is in [`docs/mcp.md`](../docs/mcp.md).
