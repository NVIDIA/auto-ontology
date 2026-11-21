<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend

This is the [Next.js](https://nextjs.org) app — the web UI **and the public API gateway** in front of the FastAPI backend. **The Python API (`auto_ontology/` at the repo root) must be running** on :3001 for API-backed routes. See the **[root README](../README.md)** for full setup, and [`frontend/CLAUDE.md`](./CLAUDE.md) for the conventions that apply under `frontend/`.

## Getting Started

From the repo root:

```bash
cp .env.example .env # then edit credentials
cd frontend && pnpm install && pnpm dev
```

`pnpm dev` pushes the Prisma schema, generates the Prisma client, and starts the Next.js dev server on port **3000**.

Start the backend separately, from the repo root:

```bash
uv run python -m auto_ontology.server # FastAPI on :3001
```

`./dev_tools/setup_env.sh --dev` brings up just the infrastructure (Postgres, pgAdmin, the ingestion service) so you don't have to start each piece by hand.

Open <http://localhost:3000>.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!
