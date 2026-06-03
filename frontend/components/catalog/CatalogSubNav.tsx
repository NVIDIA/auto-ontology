// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { HorizontalNav } from '@kui/foundations-react';

const tabs = [
	{ value: '/catalog/tables', href: '/catalog/tables', children: 'Data Dictionary' },
	{ value: '/catalog/lineage', href: '/catalog/lineage', children: 'Exploration' },
	{ value: '/catalog/tags', href: '/catalog/tags', children: 'Tags' },
	{ value: '/catalog/queries', href: '/catalog/queries', children: 'Query Tab' },
];

export const CatalogSubNav = () => {
	const pathname = usePathname();
	const active = tabs.find((t) => pathname.startsWith(t.value))?.value ?? tabs[0].value;

	return (
		<div className="border-b border-[var(--border-color-base)] bg-[var(--background-color-surface-navigation)] px-4">
			<HorizontalNav
				value={active}
				onValueChange={() => {}}
				items={tabs}
				renderLink={({ href, children, value }) => (
					<Link key={value} href={href}>
						{children}
					</Link>
				)}
			/>
		</div>
	);
};
