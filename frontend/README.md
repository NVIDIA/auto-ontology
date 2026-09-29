This is a [Next.js](https://nextjs.org) app. **The Python API in `python/` must be running** for `/data` and other API-backed routes. See the **[root README](../../README.md)** for full setup.

## Getting Started

From the repo root:

```bash
pnpm dev
```

This starts both the Next.js dev server (port **3001**) and the FastAPI backend (port **8000**) via Turborepo.

Open **http://localhost:3001**.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

run auto_ontology from root - cd auto_ontology && uv run --project .. uvicorn server.main:app --reload --host 127.0.0.1 --port 3001
