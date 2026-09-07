<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# gsf-mcp

An [MCP](https://modelcontextprotocol.io) server for
[GSF](https://github.com/NVIDIA/GSF) (Generative Semantic Fabric). It lets any
MCP-capable agent — Cursor, Claude Desktop, an internal agent — ask questions in
natural language about the data a GSF deployment is connected to, and inspect the
semantic layer behind the answers.

## Install and run

`uvx` fetches, builds, and runs it straight from the repository, so there is
nothing to clone:

```sh
uvx --from "git+https://github.com/NVIDIA/GSF.git#subdirectory=mcp" gsf-mcp
```

From a checkout of this directory, `uvx --from . gsf-mcp` does the same.

> [!NOTE]
> GSF is NVIDIA-internal today, so this install needs GitHub credentials with
> access to the repository. It becomes a plain `uvx gsf-mcp` once the package is
> published to PyPI.

Run one server; people log in through it. The URL of your GSF deployment is the
only thing it needs to be told:

```sh
GSF_API_URL=https://gsf.example.com gsf-mcp
```

## Connect a client

In Cursor (`~/.cursor/mcp.json`) or Claude Desktop
(`claude_desktop_config.json`), point at the URL. There are no credentials in the
config:

```json
{
  "mcpServers": {
    "gsf": {
      "url": "https://gsf-mcp.example/mcp"
    }
  }
}
```

The client offers to sign in, the user gets GSF's normal login page, and every
call afterwards runs as that person.

Then ask something like *"what does GSF mean by an active customer, and how many
were there last quarter?"*

## Tools

`ask_question` is the one that answers questions: it runs GSF's text-to-SQL agent
and returns the answer, the SQL it ran, and the rows. The rest — `search_terms`,
`describe_table`, `check_answerable` and friends — let an agent learn the
vocabulary and check its assumptions first.

Every tool reads, and every tool works through the semantic layer: there is none
for browsing databases, schemas, or raw columns. Nothing here modifies the
glossary, the catalog, or the underlying databases.

## Notes

This server is a plain HTTP client of the GSF API, so it needs no database
credentials and can run anywhere that can reach your deployment. It holds no
credentials of its own either — GSF signs each caller in — so it never has more
access than the person calling it.

Full documentation — all configuration variables, the complete tool list, and how
to extend it — is in [`docs/mcp.md`](../docs/mcp.md).
