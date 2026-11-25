// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { Fragment } from 'react';
import { HeaderIcon } from '@/common/HeaderIcon';
import { Text } from '@/common/Text';

/** Icons from the design system's icon library (AO-IL System), by name. */
export type CrumbIconName =
	| 'chat'
	| 'discovery'
	| 'glossary'
	| 'analysis'
	| 'analytics'
	| 'exploration'
	| 'data'
	| 'settings'
	| 'database'
	| 'schema'
	| 'table'
	| 'column';

export type BreadcrumbItem = {
	label: string;
	href?: string;
	/** Icon shown ahead of the label. */
	icon?: CrumbIconName;
};

type BreadcrumbsProps = {
	items: BreadcrumbItem[];
};

const CRUMB_MAX_WIDTH = 'max-w-64';

const CRUMB_BASE = `flex min-w-0 ${CRUMB_MAX_WIDTH} items-center gap-1 rounded py-1 pl-1 pr-2 text-sm leading-5`;

// The icons are the design library's own files, painted through a mask in the
// text color, so an icon takes the crumb's role colors (primary, secondary,
// hover green) like its label does, from one file. Each class is spelled out in
// full so Tailwind finds it. The library draws each icon at its own size inside
// a 24px slot, so the box has the file's size and the slot centers it.
const MASK_BASE = 'block shrink-0 bg-current [mask-position:center] [mask-repeat:no-repeat]';
const ICON_CLASS: Record<CrumbIconName, string> = {
	chat: 'size-6 [mask-image:url(/icons/comment.svg)]',
	// The library draws Discovery in separate pieces; the file puts them back
	// together at the positions the design gives them.
	discovery: 'size-6 [mask-image:url(/icons/discovery.svg)]',
	glossary: 'size-6 [mask-image:url(/icons/glossary.svg)]',
	analysis: 'size-6 [mask-image:url(/icons/analysis.svg)]',
	analytics: 'size-4 [mask-image:url(/icons/analytics.svg)]',
	exploration: 'size-[18px] [mask-image:url(/icons/exploration.svg)]',
	data: 'size-6 [mask-image:url(/icons/data.svg)]',
	settings: 'size-[18px] [mask-image:url(/icons/settings.svg)]',
	database: 'h-[18px] w-[14px] [mask-image:url(/icons/database.svg)]',
	schema: 'size-6 [mask-image:url(/icons/schema.svg)]',
	table: 'size-6 [mask-image:url(/icons/table.svg)]',
	column: 'size-6 [mask-image:url(/icons/column.svg)]',
};

const CrumbIcon = ({ icon }: { icon: CrumbIconName | undefined }) =>
	icon == null ? null : (
		<span aria-hidden className="flex size-6 shrink-0 items-center justify-center">
			<span className={`${MASK_BASE} ${ICON_CLASS[icon]}`} />
		</span>
	);

// Three roles, as the design names them:
// - primary: a crumb you can click, so it reads as the stronger text;
// - secondary: the page you are on, quieter because there is nowhere to go;
// - hover: a clickable crumb under the pointer, in the brand green. Its colors
//   and its icon change together, with no fade, so the icon never leads.
const CRUMB_PRIMARY = 'text-[#2b3036] dark:text-[#f0f0f3]';
const CRUMB_SECONDARY = 'text-[#5a5e64] dark:text-[#b8b8c1]';
const CRUMB_HOVER =
	'hover:rounded-lg hover:bg-[#f1f8e5] hover:text-[#76b900] dark:hover:bg-[rgba(138,210,26,0.12)] dark:hover:text-[#8ad21a]';

export const Breadcrumbs = ({ items }: BreadcrumbsProps) => {
	return (
		<nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-0.5 pl-1 pr-5">
			{items.map((item, i) => {
				const isLast = i === items.length - 1;
				const label = <Text text={item.label} />;

				return (
					<Fragment key={i}>
						{i > 0 && <HeaderIcon name="arrows" />}
						{isLast || !item.href ? (
							<span
								aria-current={isLast ? 'page' : undefined}
								className={`${CRUMB_BASE} ${isLast ? CRUMB_SECONDARY : CRUMB_PRIMARY}`}
							>
								<CrumbIcon icon={item.icon} />
								{label}
							</span>
						) : (
							<Link
								href={item.href}
								className={`group ${CRUMB_BASE} ${CRUMB_PRIMARY} ${CRUMB_HOVER}`}
							>
								<CrumbIcon icon={item.icon} />
								{label}
							</Link>
						)}
					</Fragment>
				);
			})}
		</nav>
	);
};
