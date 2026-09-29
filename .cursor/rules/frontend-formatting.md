---
description: Enforce Prettier formatting on frontend file edits
globs: frontend/**/*.{ts,tsx,js,jsx,css,json,md}
alwaysApply: false
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend Formatting — Prettier (MANDATORY)

After editing **any** file under `frontend/`, you **MUST** run Prettier before finishing:

```bash
cd frontend && npx prettier --write <relative-path-to-file>
```

Or to format all frontend files at once:

```bash
pnpm format
```

## Why this matters

- CI runs `pnpm format:check` and **will reject** unformatted code.
- Do NOT rely on manual indentation or spacing — Prettier enforces tabs, single quotes, semicolons, 100-char width, and trailing commas (see `frontend/.prettierrc`).
- Run Prettier **after every edit**, not just before committing. If you edit multiple files, format each one or run `pnpm format` once at the end.

## Common mistakes to avoid

- Writing object properties with wrong indentation depth (Prettier will fix nesting).
- Leaving long lines that exceed 100 characters (Prettier will wrap them).
- Forgetting to format after a multi-file refactor.
