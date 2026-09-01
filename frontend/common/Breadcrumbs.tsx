// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { Fragment } from 'react';
import { Icon, IconName } from '@/common/icons';
import { Text } from '@/common/Text';

export type BreadcrumbItem = {
	label: string;
	href?: string;
};

type BreadcrumbsProps = {
	items: BreadcrumbItem[];
};

const CRUMB_MAX_WIDTH = 'max-w-64';

export const Breadcrumbs = ({ items }: BreadcrumbsProps) => {
	return (
		<nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1 text-sm">
			{items.map((item, i) => {
				const isLast = i === items.length - 1;

				return (
					<Fragment key={i}>
						{i > 0 && (
							<Icon
								name={IconName.ChevronRight}
								className="h-3.5 w-3.5 shrink-0 text-zinc-400 dark:text-zinc-600"
							/>
						)}
						{isLast || !item.href ? (
							<span
								className={`min-w-0 ${CRUMB_MAX_WIDTH} text-zinc-900 dark:text-zinc-100`}
							>
								<Text text={item.label} />
							</span>
						) : (
							<Link
								href={item.href}
								className={`min-w-0 ${CRUMB_MAX_WIDTH} text-zinc-500 transition-colors hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100`}
							>
								<Text text={item.label} />
							</Link>
						)}
					</Fragment>
				);
			})}
		</nav>
	);
};
