// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';

// Better Auth mounts its whole surface behind this one catch-all
// (`toNextJsHandler`), so the concrete sub-paths — sign-in/out, session, the
// SSO authorize/callback pair — are the library's contract rather than ours,
// and there is no per-endpoint shape to declare. What this pins down is that
// the segment exists, is unauthenticated by nature, and answers JSON.
const responses = {
	200: {
		description:
			'Handled by Better Auth; the body depends on `{all}`. See ' +
			'https://better-auth.com/docs for the per-endpoint contracts.',
		schema: z.record(z.string(), z.unknown()),
	},
	404: { description: '`{all}` is not a Better Auth route.' },
};

export const openapi: OpenApiRoute = {
	get: { responses },
	post: { responses },
};
