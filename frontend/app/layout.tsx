// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Metadata } from 'next';
import { headers } from 'next/headers';
import { Geist, Geist_Mono } from 'next/font/google';
import { TooltipProvider } from '@nvidia/foundations-react-core';
import { NavRail } from '@/common/NavRail';
import { AppTopBar } from '@/common/AppTopBar';
import { BreadcrumbProvider } from '@/contexts/BreadcrumbContext';
import { getCurrentSession } from '@/auth/auth-guards';
import { Role } from '@/enums/auth';
import './globals.css';

const isLoginOrOauthPath = (pathname: string): boolean =>
	pathname === '/login' ||
	pathname.startsWith('/login/') ||
	pathname === '/oauth' ||
	pathname.startsWith('/oauth/');

const geistSans = Geist({
	variable: '--font-geist-sans',
	subsets: ['latin'],
});

const geistMono = Geist_Mono({
	variable: '--font-geist-mono',
	subsets: ['latin'],
});

export const metadata: Metadata = {
	title: {
		default: 'Auto Ontology — NVIDIA',
		template: 'Auto Ontology - %s',
	},
	description: 'NVIDIA Auto Ontology',
	icons: {
		icon: '/favicon.svg',
	},
};

export default async function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	const session = await getCurrentSession();
	const isAdmin = session?.user.role === Role.Admin;
	// Injected at runtime by the Helm chart from the chart version (.Chart.Version).
	const appVersion = process.env.APP_VERSION;
	const pathname = (await headers()).get('x-auto-ontology-pathname') ?? '';
	const showAppChrome = Boolean(session) && !isLoginOrOauthPath(pathname);

	return (
		<html
			lang="en"
			className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
		>
			<body className="flex h-full flex-col bg-white text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
				<TooltipProvider openDelayDuration={400} skipDelayDuration={0}>
					{showAppChrome ? (
						<BreadcrumbProvider>
							<AppTopBar version={appVersion} />
							<div className="flex min-h-0 flex-1">
								<NavRail isAdmin={isAdmin} />
								<div className="min-w-0 flex-1">{children}</div>
							</div>
						</BreadcrumbProvider>
					) : (
						children
					)}
				</TooltipProvider>
			</body>
		</html>
	);
}
