// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The fields each Prisma-backed route publishes.
 *
 * These handlers answer with the row itself (`NextResponse.json(row)`), so
 * without an explicit `select` the API is "whatever the model happens to
 * have" — add a column and it ships, undeclared and unreviewed. Naming every
 * field here keeps the exposed surface a deliberate choice, and Prisma narrows
 * the returned type to match, so the zod declarations in the neighbouring
 * `openapi.ts` modules describe what actually goes out.
 *
 * Each list mirrors a schema in `lib/openapiSchemas.ts`; change one and change
 * the other.
 */

export const conversationSelect = {
	id: true,
	user_id: true,
	title: true,
	created_at: true,
	updated_at: true,
} as const;

export const messageSelect = {
	id: true,
	conversation_id: true,
	role: true,
	content: true,
	sql_code: true,
	sql_response: true,
	created_at: true,
} as const;

export const acronymSelect = {
	id: true,
	name: true,
	description: true,
	created_at: true,
	updated_at: true,
} as const;

/** The analytics report joins the User to resolve a display name. */
export const analyticsSelect = {
	id: true,
	user_id: true,
	source: true,
	question: true,
	question_timestamp: true,
	response: true,
	response_timestamp: true,
	sql: true,
	user: { select: { id: true, name: true, email: true, role: true } },
} as const;
