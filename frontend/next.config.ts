// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import path from 'node:path';
import dotenv from 'dotenv';
import type { NextConfig } from 'next';

dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

// NOTE: there are intentionally no `rewrites()`. Every /api/* path that used to
// be proxied to the Python backend via a rewrite is now served by a dedicated,
// permission-gated route handler under app/api/** that forwards to the backend
// (see auth/proxy-backend.ts + withPermission). Rewrites bypass route handlers,
// so they can't be access-controlled — hence the move.
const nextConfig: NextConfig = {
	output: process.env.NEXT_OUTPUT === 'standalone' ? 'standalone' : undefined,
	outputFileTracingRoot: path.resolve(__dirname, '..'),
	transpilePackages: ['@nvidia/foundations-react-core'],
	turbopack: {
		rules: {
			'*.svg': {
				loaders: [{ loader: '@svgr/webpack', options: { svgo: false } }],
				as: '*.js',
			},
		},
	},
};

export default nextConfig;
