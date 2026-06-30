// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import path from 'node:path';
import dotenv from 'dotenv';
import type { NextConfig } from 'next';

dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

const pythonApiUrl = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

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
	async rewrites() {
		return [
			{
				source: '/api/chat/:path*',
				destination: `${pythonApiUrl}/api/chat/:path*`,
			},
			{
				source: '/api/datasources/:path*',
				destination: `${pythonApiUrl}/api/datasources/:path*`,
			},
			{
				source: '/api/schemas/:path*',
				destination: `${pythonApiUrl}/api/schemas/:path*`,
			},
			{
				source: '/api/tables/:path*',
				destination: `${pythonApiUrl}/api/tables/:path*`,
			},
			{
				source: '/api/columns/:path*',
				destination: `${pythonApiUrl}/api/columns/:path*`,
			},
			{
				source: '/api/nodes/:path*',
				destination: `${pythonApiUrl}/api/nodes/:path*`,
			},
			{
				source: '/api/custom-analyses',
				destination: `${pythonApiUrl}/api/custom-analyses`,
			},
			{
				source: '/api/custom-analyses/:path*',
				destination: `${pythonApiUrl}/api/custom-analyses/:path*`,
			},
			{
				source: '/api/connections/:path*',
				destination: `${pythonApiUrl}/api/connections/:path*`,
			},
			{
				source: '/api/zones/:path*',
				destination: `${pythonApiUrl}/api/zones/:path*`,
			},
			{
				source: '/api/terms/:path*',
				destination: `${pythonApiUrl}/api/terms/:path*`,
			},
			{
				source: '/api/users/:path*',
				destination: `${pythonApiUrl}/api/users/:path*`,
			},
			{
				source: '/api/health',
				destination: `${pythonApiUrl}/api/health`,
			},
		];
	},
};

export default nextConfig;
