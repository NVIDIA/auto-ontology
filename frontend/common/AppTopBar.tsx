// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Icon, IconName } from '@/common/icons';
import { Breadcrumbs } from '@/common/Breadcrumbs';
import { GlobalSearch } from '@/common/GlobalSearch';
import { UserMenu } from '@/common/UserMenu';
import { useBreadcrumbs } from '@/contexts/BreadcrumbContext';

export const AppTopBar = ({ version }: { version?: string }) => {
	const { items, rightSlot } = useBreadcrumbs();

	return (
		<header className="flex h-[52px] shrink-0 items-center border-b border-zinc-200 bg-white px-4 pr-[40px] dark:border-zinc-800 dark:bg-zinc-950">
			<Icon name={IconName.NvidiaLogo} className="mr-3 h-5 w-5 shrink-0" />
			<Breadcrumbs items={items} />
			<div className="ml-auto flex items-center gap-3">
				{rightSlot ? <div className="flex items-center gap-2">{rightSlot}</div> : null}
				<GlobalSearch />
				<UserMenu version={version} />
			</div>
		</header>
	);
};
