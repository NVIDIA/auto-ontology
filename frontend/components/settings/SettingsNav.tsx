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
const ZONES_NAV_ITEM: NavItem = { label: 'Zones', href: '/settings/zones' };
const USERS_NAV_ITEM: NavItem = { label: 'Users', href: '/settings/users' };
const SSO_NAV_ITEM: NavItem = { label: 'Single Sign-On', href: '/settings/sso' };
const SEMANTIC_COMPILATION_NAV_ITEM: NavItem = {
	label: 'Semantic Compilation',
	href: '/settings/semantic-compilation',
};
const AGENT_SETTINGS_NAV_ITEM: NavItem = {
	label: 'Agent Settings',
	href: '/settings/agent-settings',
};
const IMPORT_EXPORT_NAV_ITEM: NavItem = {
	label: 'Import / Export',
	href: '/settings/import-export',
};

function rowClassName(selected: boolean) {
	return `flex min-h-9 items-center rounded-lg px-3 text-sm no-underline transition-colors ${
		selected
			? 'bg-[#76b900]/15 font-medium text-zinc-900 shadow-sm ring-1 ring-[#76b900]/30 dark:text-zinc-100'
			: 'text-zinc-800 hover:bg-zinc-100/90 dark:text-zinc-200 dark:hover:bg-zinc-800/70'
	}`;
}

// The whole Settings section is admin-only (gated in app/settings/layout.tsx),
// so no per-item role checks are needed here.
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

	const items: NavItem[] = [];
	if (!envManaged) items.push(CONNECTIONS_NAV_ITEM);
	items.push(
		ZONES_NAV_ITEM,
		USERS_NAV_ITEM,
		SSO_NAV_ITEM,
		SEMANTIC_COMPILATION_NAV_ITEM,
		AGENT_SETTINGS_NAV_ITEM,
		IMPORT_EXPORT_NAV_ITEM,
	);

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
