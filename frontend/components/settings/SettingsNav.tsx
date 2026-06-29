// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { connectionsApi } from '@/api/connections';

type NavItem = { label: string; href: string };

const CONNECTIONS_NAV_ITEM: NavItem = { label: 'Connections', href: '/settings/connections' };
const SEMANTIC_INPUT_NAV_ITEM: NavItem = {
	label: 'Semantic Input',
	href: '/settings/semantic-input',
};
const ZONES_NAV_ITEM: NavItem = {
	label: 'Zones',
	href: '/settings/zones',
};

export const SETTINGS_NAV_ITEMS: readonly NavItem[] = [
	CONNECTIONS_NAV_ITEM,
	SEMANTIC_INPUT_NAV_ITEM,
	ZONES_NAV_ITEM,
];

function rowClassName(selected: boolean) {
	return `flex min-h-9 items-center rounded-lg px-3 text-sm no-underline transition-colors ${
		selected
			? 'bg-[#76b900]/15 font-medium text-zinc-900 shadow-sm ring-1 ring-[#76b900]/30 dark:text-zinc-100'
			: 'text-zinc-800 hover:bg-zinc-100/90 dark:text-zinc-200 dark:hover:bg-zinc-800/70'
	}`;
}

export const SettingsNav = () => {
	const pathname = usePathname();
	// Hide the Connections section when connections are managed via the
	// CONNECTION_STRINGS env var (they are fixed by config, not editable here).
	const [envManaged, setEnvManaged] = useState(false);

	useEffect(() => {
		let active = true;
		void (async () => {
			const res = await connectionsApi.isEnvSource();
			if (active && typeof res === 'boolean') {
				setEnvManaged(res);
			}
		})();
		return () => {
			active = false;
		};
	}, []);

	const items = envManaged ? [SEMANTIC_INPUT_NAV_ITEM, ZONES_NAV_ITEM] : SETTINGS_NAV_ITEMS;

	return (
		<nav
			className="flex min-h-0 flex-1 flex-col gap-y-1 overflow-y-auto px-3 py-2 sm:px-4"
			aria-label="Settings sections"
		>
			{items.map((item) => {
				const selected = pathname === item.href || pathname.startsWith(`${item.href}/`);
				return (
					<Link key={item.href} href={item.href} className={rowClassName(selected)}>
						{item.label}
					</Link>
				);
			})}
		</nav>
	);
};
