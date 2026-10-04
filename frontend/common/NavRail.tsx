// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Icon, IconName } from '@/common/icons';

type NavItem = {
	icon: IconName;
	href: string;
	label: string;
	adminOnly?: boolean;
};

const navItems: NavItem[] = [
	{ icon: IconName.ChatBubble, href: '/chat', label: 'Chat' },
	{ icon: IconName.Terms, href: '/terms', label: 'Terms' },
	{ icon: IconName.ChartBar, href: '/analysis', label: 'Analysis' },
	{ icon: IconName.Exploration, href: '/exploration', label: 'Exploration' },
	{ icon: IconName.Database, href: '/data', label: 'Data' },
	{ icon: IconName.ChartLine, href: '/analytics', label: 'Analytics', adminOnly: true },
	// Settings is admin-only and now also contains Users and Single Sign-On
	// (see app/settings/*). Viewers don't see it.
	{ icon: IconName.Settings, href: '/settings', label: 'Settings', adminOnly: true },
];

export const NavRail = ({ isAdmin = false }: { isAdmin?: boolean }) => {
	const pathname = usePathname();
	const [isExpanded, setIsExpanded] = useState(false);
	const visibleItems = navItems.filter((item) => !item.adminOnly || isAdmin);

	return (
		<nav className="relative h-full w-12 shrink-0 border-r border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
			<div
				onMouseEnter={() => setIsExpanded(true)}
				onMouseLeave={() => setIsExpanded(false)}
				className={`absolute inset-y-0 left-0 flex flex-col overflow-hidden bg-white py-3 transition-[width] duration-200 ease-out dark:bg-zinc-950 ${
					isExpanded
						? 'z-30 w-56 border-r border-zinc-200 shadow-lg dark:border-zinc-800'
						: 'w-full'
				}`}
			>
				<div
					className={`flex flex-1 flex-col gap-2 ${isExpanded ? 'px-1.5' : 'items-center'}`}
				>
					{visibleItems.map((item) => {
						const isActive = pathname.startsWith(item.href);

						return (
							<Link
								key={item.href}
								href={item.href}
								title={isExpanded ? undefined : item.label}
								className={`flex h-9 shrink-0 items-center gap-3 rounded-lg transition-colors ${
									isExpanded ? 'w-full px-2' : 'w-9 justify-center'
								} ${
									isActive
										? 'bg-[#76b900]/15 text-[#76b900]'
										: 'text-body hover:bg-zinc-100 hover:text-heading dark:hover:bg-zinc-800 dark:hover:text-zinc-200'
								}`}
							>
								<Icon name={item.icon} className="h-5 w-5 shrink-0" />
								{isExpanded && (
									<span className="truncate text-sm font-medium whitespace-nowrap">
										{item.label}
									</span>
								)}
							</Link>
						);
					})}
				</div>
			</div>
		</nav>
	);
};
