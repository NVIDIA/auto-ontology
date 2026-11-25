// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { Fragment } from 'react';
import { HeaderIcon } from '@/common/HeaderIcon';
import { Text } from '@/common/Text';

export type BreadcrumbItem = {
	label: string;
	href?: string;
	/** Shows the database icon ahead of the label, as the design does for connectors. */
	database?: boolean;
};

type BreadcrumbsProps = {
	items: BreadcrumbItem[];
};

const CRUMB_MAX_WIDTH = 'max-w-64';

const CRUMB_BASE = `flex min-w-0 ${CRUMB_MAX_WIDTH} items-center gap-1 rounded py-1 pl-1 pr-2 text-sm leading-5`;

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
								{item.database ? (
									<HeaderIcon name={isLast ? 'database-current' : 'database'} />
								) : null}
								{label}
							</span>
						) : (
							<Link
								href={item.href}
								className={`group ${CRUMB_BASE} ${CRUMB_PRIMARY} ${CRUMB_HOVER}`}
							>
								{item.database ? (
									<>
										<span className="group-hover:hidden">
											<HeaderIcon name="database" />
										</span>
										<span className="hidden group-hover:block">
											<HeaderIcon name="database-selected" />
										</span>
									</>
								) : null}
								{label}
							</Link>
						)}
					</Fragment>
				);
			})}
		</nav>
	);
};
