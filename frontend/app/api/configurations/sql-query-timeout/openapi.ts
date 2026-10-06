// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import {
	DEFAULT_SQL_QUERY_TIMEOUT_SECONDS,
	MAX_SQL_QUERY_TIMEOUT_SECONDS,
	MIN_SQL_QUERY_TIMEOUT_SECONDS,
} from '@/lib/sqlQueryTimeout';
import type { OpenApiRoute } from '@/types/openapi';

// Exported so route.ts enforces exactly what this spec advertises, rather than
// the two drifting apart.
export const sqlQueryTimeoutBody = z.object({
	seconds: z.number().int().min(MIN_SQL_QUERY_TIMEOUT_SECONDS).max(MAX_SQL_QUERY_TIMEOUT_SECONDS),
});

const sqlQueryTimeoutSchema = z.object({
	seconds: z.number().int().describe('Statement timeout in seconds.'),
});

export const openapi: OpenApiRoute = {
	get: {
		responses: {
			200: {
				description:
					'How long a SQL statement issued by the text-to-SQL agent may run before ' +
					`it is cancelled. ${DEFAULT_SQL_QUERY_TIMEOUT_SECONDS} when unset.`,
				schema: sqlQueryTimeoutSchema,
			},
		},
	},
	put: {
		body: {
			description: `\`seconds\` must be an integer from ${MIN_SQL_QUERY_TIMEOUT_SECONDS} to ${MAX_SQL_QUERY_TIMEOUT_SECONDS}.`,
			schema: sqlQueryTimeoutBody,
		},
		responses: {
			200: {
				description: 'The stored timeout. Applies from the next statement.',
				schema: sqlQueryTimeoutSchema,
			},
			400: { description: 'Body is not JSON, or `seconds` is not an in-range integer.' },
		},
	},
};
