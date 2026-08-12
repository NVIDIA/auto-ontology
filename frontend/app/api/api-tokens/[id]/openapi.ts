// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

export const openapi: OpenApiRoute = {
	delete: {
		sessionOnly: true,
		path: z.object({ id: z.string().describe('Id of one of the caller’s own tokens.') }),
		responses: {
			200: {
				description: 'The token is revoked; requests carrying it now fail with 401.',
				schema: z.object({ id: z.string() }),
			},
			403: {
				description:
					'The caller authenticated with an API token — revocation requires a session.',
			},
			404: { description: 'No such token belongs to the caller.' },
		},
	},
};
