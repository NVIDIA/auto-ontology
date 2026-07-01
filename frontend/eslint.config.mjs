// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, globalIgnores } from 'eslint/config';
import nextVitals from 'eslint-config-next/core-web-vitals';
import nextTs from 'eslint-config-next/typescript';

const HTTP_METHODS = 'GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS';

// Enforce that every API route handler is wrapped in an auth decorator
// (`withPublic` or `withPermission` from `@/auth/with-auth`) — never a bare
// handler. This makes "public vs permission-checked" an auditable, build-time
// guarantee: a new route that forgets a guard fails `next build` / `next lint`.
// Excludes app/api/auth/** (Better Auth's own `toNextJsHandler` export).
const routeAuthCoverage = {
	files: ['app/api/**/route.ts'],
	ignores: ['app/api/auth/**'],
	rules: {
		'no-restricted-syntax': [
			'error',
			{
				selector: `ExportNamedDeclaration > FunctionDeclaration[id.name=/^(${HTTP_METHODS})$/]`,
				message:
					'API route handlers must be wrapped: `export const GET = withPublic(fn)` or `withPermission({...})(fn)`, not a bare `export async function GET`.',
			},
			{
				selector: `ExportNamedDeclaration > VariableDeclaration > VariableDeclarator[id.name=/^(${HTTP_METHODS})$/]:not([init.callee.name='withPublic']):not([init.callee.callee.name='withPermission'])`,
				message:
					'API route handlers must be wrapped in withPublic(...) or withPermission({...})(...) from @/auth/with-auth.',
			},
		],
	},
};

const eslintConfig = defineConfig([
	...nextVitals,
	...nextTs,
	routeAuthCoverage,
	globalIgnores(['.next/**', 'out/**', 'build/**', 'next-env.d.ts']),
]);

export default eslintConfig;
