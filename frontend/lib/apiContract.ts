// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Compile-time proof that the Prisma-backed routes publish what they declare.
 *
 * `lib/apiSelects.ts` decides which columns leave the handler;
 * `lib/openapiSchemas.ts` (and the analytics `openapi.ts`) describe them in the
 * OpenAPI spec. Those are two lists that have to agree, and nothing forced them
 * to — until this file. Each assertion below fails the build when a select
 * gains, loses or retypes a field without the schema following, so the spec
 * cannot quietly drift from the responses.
 *
 * This module is types only; it emits no runtime code.
 */

import type { Prisma } from '@/generated/prisma/client';
import type { z } from 'zod';
import type { analyticsRow } from '@/app/api/analytics/openapi';
import type {
	acronymSelect,
	analyticsSelect,
	conversationSelect,
	messageSelect,
} from './apiSelects';
import type { acronymSchema, conversationSchema, messageSchema } from './openapiSchemas';

/**
 * What a row looks like once `NextResponse.json` has been through it: Prisma
 * hands back `Date` objects, the wire carries ISO strings.
 */
type Serialized<T> = T extends Date
	? string
	: T extends (infer U)[]
		? Serialized<U>[]
		: T extends object
			? { [K in keyof T]: Serialized<T[K]> }
			: T;

/** `true` only when the two types are mutually assignable. */
type Exact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false;

// A failure here reads as "Type 'false' is not assignable to type 'true'" on
// the line naming the model that drifted.
const conversationMatches: Exact<
	Serialized<Prisma.ConversationGetPayload<{ select: typeof conversationSelect }>>,
	z.infer<typeof conversationSchema>
> = true;

const messageMatches: Exact<
	Serialized<Prisma.MessageGetPayload<{ select: typeof messageSelect }>>,
	z.infer<typeof messageSchema>
> = true;

const acronymMatches: Exact<
	Serialized<Prisma.AcronymGetPayload<{ select: typeof acronymSelect }>>,
	z.infer<typeof acronymSchema>
> = true;

/**
 * The analytics handler is the one that transforms its row: `User.role` is
 * nullable in the database and the report substitutes `viewer`, so the
 * published type is narrower than the select. Spelling that out here keeps the
 * exception visible instead of weakening the check for every other field.
 */
type AnalyticsPayload = Serialized<
	Prisma.ConversationAnalyticsGetPayload<{ select: typeof analyticsSelect }>
>;
type AnalyticsPublished = Omit<AnalyticsPayload, 'user'> & {
	user: Omit<AnalyticsPayload['user'], 'role'> & { role: string };
};

const analyticsMatches: Exact<AnalyticsPublished, z.infer<typeof analyticsRow>> = true;

export type { Exact, Serialized };
export { acronymMatches, analyticsMatches, conversationMatches, messageMatches };
