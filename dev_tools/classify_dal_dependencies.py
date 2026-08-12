# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which DAL functions read the semantic tier, and by what route.

Phase 6 needed this to decide what it could actually verify: nothing writes
terms or attributes until Phase 7, so a function that joins to them cannot be
checked before then — against no data, a wrong join is indistinguishable from a
correct one.

**Committed because three hand-analyses got it wrong**, each more confidently
than the last: 21/5, then 20/6, then 19/7. The first used a regex that stopped
at the first line beginning with a letter, and every query constant here is a
Cypher string whose second line is ``MATCH``. The second recursed but guarded
self-reference with "the text does not start with its own name", which is true
of every assignment, so the recursion never ran and it reproduced the first
answer. The third was hand-checking, which found two more but had no reason to
stop where it did.

This resolves constants transitively through AST line ranges with a real
visited-set, and prints the chain that proves each verdict so the answer can be
checked rather than trusted.

Usage::

    uv run --no-sync python -m dev_tools.classify_dal_dependencies [module]
"""

import ast
import pathlib

import sys

MODULE = sys.argv[1] if len(sys.argv) > 1 else "datasources"
SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "gsf"
    / "dal"
    / "neo4j"
    / f"{MODULE}.py"
)
if not SRC.is_file():
    raise SystemExit(f"no such module: {SRC}")
text = SRC.read_text()
lines = text.split("\n")
tree = ast.parse(text)

consts: dict[str, str] = {}
funcs: dict[str, str] = {}
for node in tree.body:
    body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                consts[target.id] = body
    elif isinstance(node, ast.FunctionDef):
        funcs[node.name] = body

SEMANTIC = (
    "column_description_expr",
    "table_description_expr",
    "LABEL_TERM",
    "LABEL_COLUMN_ATTRIBUTE",
    "LABEL_SQL_ATTRIBUTE",
    "REL_REPRESENTS",
    "REL_HAS_ATTRIBUTE",
    "REL_SEMANTIC_FK",
    "REL_PROPERTY_OF",
    "SQL_ATTR_SOURCE_BRIDGE",
    "TABLE_COUNTS_SUBQUERY",
)


def reads_semantic(name: str, body: str, seen: set[str]) -> list[str]:
    """Return the chain of names proving *body* reaches the semantic tier."""
    for marker in SEMANTIC:
        if marker in body:
            return [name, marker]
    for const_name, const_body in consts.items():
        if const_name in seen or const_name == name:
            continue
        # Reference, not the definition line itself.
        if const_name not in body:
            continue
        chain = reads_semantic(const_name, const_body, seen | {const_name})
        if chain:
            return [name, *chain]
    # Local helper functions count too.
    for fn_name, fn_body in funcs.items():
        if fn_name in seen or fn_name == name or not fn_name.startswith("_"):
            continue
        if fn_name not in body:
            continue
        chain = reads_semantic(fn_name, fn_body, seen | {fn_name})
        if chain:
            return [name, *chain]
    return []


pure, semantic = [], []
for fn_name, body in funcs.items():
    if fn_name.startswith("_"):
        continue
    chain = reads_semantic(fn_name, body, {fn_name})
    (semantic if chain else pure).append((fn_name, chain))

print(f"CATALOG ONLY ({len(pure)}):")
for name, _ in sorted(pure):
    print(f"   {name}")
print(f"\nREADS SEMANTIC TIER ({len(semantic)}):")
for name, chain in sorted(semantic):
    print(f"   {name}")
    print(f"       via {' -> '.join(chain[1:])}")
