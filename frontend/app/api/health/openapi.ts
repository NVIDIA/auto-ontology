// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

const check = z.union([
	z.object({ status: z.literal('ok') }),
	z
		.object({ status: z.literal('error'), detail: z.string() })
		.describe('`detail` is the driver error, truncated to 200 characters.'),
]);

const health = (status: 'ok' | 'degraded') =>
	z.object({ status: z.literal(status), postgres: check });

export const openapi: OpenApiRoute = {
	get: {
		responses: {
			200: { description: 'Postgres answered a `SELECT 1`.', schema: health('ok') },
			503: {
				description: 'Postgres is unreachable; the probe body carries the reason.',
				schema: health('degraded'),
			},
		},
	},
};
