// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { AUTHORS_PARAM, AUTHORS_PARAM_ON } from '@/constants/tags';
import type { OpenApiRoute } from '@/types/openapi';

/**
 * The same `authors` switch the tag list takes, for the same reason — see
 * `app/api/tags/openapi.ts`. Both GETs run it through `authorsRequested`, so a
 * client that learns the parameter from one route may use it on the other.
 */
export const openapi: OpenApiRoute = {
	get: {
		query: z.object({
			[AUTHORS_PARAM]: z
				.literal(AUTHORS_PARAM_ON)
				.optional()
				.describe(
					'Resolve `created_by` / `modified_by` to the accounts they name, adding ' +
						'`created_by_user` and `modified_by_user` beside them. Off by default: this ' +
						"view renders a tag's name and the objects carrying it, and reads no author.",
				),
		}),
	},
};
