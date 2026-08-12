// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Contract for the `openapi.ts` modules that sit next to a `route.ts`.
 *
 * `scripts/generate-openapi.ts` merges request/response schemas from
 * `docs/openapi/backend.json` for handlers that proxy to FastAPI. Handlers with no
 * FastAPI counterpart (the Prisma-backed ones) and handlers that build their
 * own upstream request (the chat routes) have nothing to merge, so they declare
 * their shapes here instead.
 *
 * The declaration lives in a *sibling* module rather than in `route.ts` because
 * the generator is a plain Node script: importing `route.ts` pulls in
 * `next/headers`, `after()` and `getPrisma()`, none of which work outside the
 * Next runtime. `openapi.ts` imports nothing but zod, so it imports cleanly.
 *
 *   // app/api/acronyms/openapi.ts
 *   export const openapi: OpenApiRoute = {
 *     get: { responses: { 200: { description: '…', schema: z.array(acronym) } } },
 *   };
 */

import type { z } from 'zod';

/** One response, keyed by status code in `OpenApiOperation.responses`. */
export type OpenApiResponse = {
	description: string;
	/** Omit for a body-less response (204, or a bare status relay). */
	schema?: z.ZodType;
	/** Defaults to `application/json`. */
	contentType?: string;
};

export type OpenApiRequestBody = {
	description?: string;
	schema: z.ZodType;
	/** Defaults to `true`. */
	required?: boolean;
	/** Defaults to `application/json`. */
	contentType?: string;
};

export type OpenApiOperation = {
	/** One query parameter per property; optional properties are not required. */
	query?: z.ZodObject;
	/**
	 * Descriptions/schemas for the path parameters this route's folder names.
	 *
	 * Optional: every `{param}` in the path is declared automatically as a
	 * required string, since OpenAPI rejects a path template whose parameters
	 * are undeclared. Use this only to add a description or a tighter schema.
	 */
	path?: z.ZodObject;
	body?: OpenApiRequestBody;
	/**
	 * Keyed by status code. A declared status replaces the one inherited from
	 * `docs/openapi/backend.json`; `null` removes an inherited status the Next.js
	 * handler can never actually return.
	 */
	responses?: Record<string, OpenApiResponse | null>;
	/**
	 * Set when the handler additionally requires a browser session, so an API
	 * token is not accepted even though the route is permission-gated (see
	 * `requireSessionCaller`). Publishes the `SessionCookie` scheme instead of
	 * `ApiToken`, so a generated client is not told to send a credential the
	 * route will reject.
	 */
	sessionOnly?: boolean;
};

export type OpenApiHttpMethod = 'get' | 'post' | 'put' | 'patch' | 'delete' | 'head' | 'options';

/** The `openapi` export of an `app/api/**\/openapi.ts` module. */
export type OpenApiRoute = Partial<Record<OpenApiHttpMethod, OpenApiOperation>>;
