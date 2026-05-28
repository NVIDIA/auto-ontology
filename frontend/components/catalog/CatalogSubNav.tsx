// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

type Tab = { href: string; label: string; matchPrefix?: string };

const tabs: Tab[] = [
	{ href: '/catalog', label: 'Overview', matchPrefix: '/catalog' },
	{ href: '/catalog/tables', label: 'Data Dictionary' },
	{ href: '/catalog/tags', label: 'Tags' },
	{ href: '/catalog/queries', label: 'Query Tab' },
];

export const CatalogSubNav = () => {
	const pathname = usePathname();
	return (
		<nav className="flex items-center gap-1 border-b border-zinc-200 bg-white px-6 dark:border-zinc-800 dark:bg-zinc-950">
			{tabs.map((t) => {
				const isActive =
					t.label === 'Overview' ? pathname === t.href : pathname.startsWith(t.href);
				return (
					<Link
						key={t.href}
						href={t.href}
						className={`relative px-3 py-3 text-sm font-medium transition-colors ${
							isActive
								? 'text-[#76b900]'
								: 'text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-200'
						}`}
					>
						{t.label}
						{isActive ? (
							<span className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-[#76b900]" />
						) : null}
					</Link>
				);
			})}
		</nav>
	);
};
