// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Zod shapes shared by the `app/api/**\/openapi.ts` declarations.
 *
 * These mirror what the Prisma-backed handlers actually serialise — a Prisma
 * row goes through `NextResponse.json`, so `DateTime` columns arrive as ISO
 * strings and nullable columns as `null`, not `undefined`. They are documented
 * by hand rather than generated from `prisma/schema.prisma` because the API
 * shapes and the table shapes are not the same thing: handlers project, join
 * (`analytics` embeds the User) and re-key, and a generator would publish
 * columns the API never returns.
 *
 * This module is imported by the OpenAPI generator, so it must stay free of
 * Next-runtime imports (`next/headers`, `getPrisma()`, …).
 */

import { z } from 'zod';

/** `format` alone, without zod's very long uuid/date-time validation patterns. */
const uuid = () => z.string().meta({ format: 'uuid' });
const timestamp = () => z.string().meta({ format: 'date-time' });

/** `NextResponse.json({ error }, { status })`, the shape every handler uses. */
export const errorSchema = z.object({ error: z.string() });

export const conversationSchema = z.object({
	id: uuid(),
	user_id: z.string().describe('Owner of the conversation; always the caller.'),
	title: z.string(),
	created_at: timestamp(),
	updated_at: timestamp(),
});

export const messageSchema = z.object({
	id: uuid(),
	conversation_id: uuid(),
	role: z.string().describe('`user` or `assistant`.'),
	content: z.string(),
	sql_code: z.string().nullable().describe('SQL the assistant produced, when it produced any.'),
	sql_response: z
		.string()
		.nullable()
		.describe('Serialised result table / chart payload for the bubble.'),
	created_at: timestamp(),
});

export const promptSchema = z.object({
	id: uuid(),
	content: z.string(),
});

export const acronymSchema = z.object({
	id: uuid(),
	name: z.string(),
	description: z.string(),
	created_at: timestamp(),
	updated_at: timestamp(),
});

/** The instance-wide on/off settings under `/api/configurations/*`. */
export const enabledFlagSchema = z.object({ enabled: z.boolean() });
