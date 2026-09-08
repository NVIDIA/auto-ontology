// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { AUTHORS_PARAM, AUTHORS_PARAM_ON } from '@/constants/tags';
import type { OpenApiRoute } from '@/types/openapi';

/**
 * Only the `authors` switch is declared here.
 *
 * Everything else on this route — bodies, responses, the tag schema itself — is
 * merged from the backend spec, which is where those live. This parameter has no
 * backend counterpart to merge: FastAPI answers with the ids it stored, and
 * turning them into names happens in the route handler, so nothing would
 * publish it if this file did not.
 */
export const openapi: OpenApiRoute = {
	get: {
		query: z.object({
			[AUTHORS_PARAM]: z
				.literal(AUTHORS_PARAM_ON)
				.optional()
				.describe(
					'Resolve `created_by` / `modified_by` to the accounts they name, adding ' +
						'`created_by_user` and `modified_by_user` beside them. Ignored for a caller ' +
						'who may not manage tags, and left off by default: the tag picker reads this ' +
						'list on every detail page and has nothing to show an author in.',
				),
		}),
	},
};
