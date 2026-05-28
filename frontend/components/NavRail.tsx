// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Icon, IconName } from '@/components/icons';

type NavItem = {
	icon: IconName;
	href: string;
	label: string;
};

const navItems: NavItem[] = [
	{ icon: IconName.ChatBubble, href: '/chat', label: 'Chat' },
	{ icon: IconName.ChartBar, href: '/analysis', label: 'Analysis' },
	{ icon: IconName.Database, href: '/data', label: 'Data Catalog' },
	{ icon: IconName.Sparkles, href: '/catalog', label: 'OpenMetadata Catalog' },
	{ icon: IconName.Settings, href: '/settings', label: 'Settings' },
];

export const NavRail = () => {
	const pathname = usePathname();

	return (
		<nav className="flex h-full w-12 shrink-0 flex-col items-center border-r border-zinc-200 bg-white py-3 dark:border-zinc-800 dark:bg-zinc-950">
			<div className="flex flex-1 flex-col items-center gap-2">
				{navItems.map((item) => {
					const isActive = pathname.startsWith(item.href);

					return (
						<Link
							key={item.href}
							href={item.href}
							title={item.label}
							className={`flex h-9 w-9 items-center justify-center rounded-lg transition-colors ${
								isActive
									? 'bg-[#76b900]/15 text-[#76b900]'
									: 'text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600 dark:hover:bg-zinc-800 dark:hover:text-zinc-200'
							}`}
						>
							<Icon name={item.icon} className="h-5 w-5" />
						</Link>
					);
				})}
			</div>
		</nav>
	);
};
