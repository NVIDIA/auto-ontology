// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Breadcrumbs } from '@/common/Breadcrumbs';
import { GlobalSearch } from '@/common/GlobalSearch';
import { UserMenu } from '@/common/UserMenu';
import { useBreadcrumbs } from '@/contexts/BreadcrumbContext';

export const AppTopBar = ({ version }: { version?: string }) => {
	const { items, rightSlot } = useBreadcrumbs();

	return (
		<header className="flex h-[52px] font-sf shrink-0 items-center justify-between border-b border-[#e9eaea] bg-white pl-3 pr-5 dark:border-[#2e2e33] dark:bg-[#1b1b1d]">
			<div className="flex min-w-0 items-center gap-3">
				<div className="flex size-9 shrink-0 items-center justify-center">
					{/* Plain <img>: a static, fixed-size logo that needs no optimization. */}
					{/* eslint-disable-next-line @next/next/no-img-element */}
					<img src="/header/nvidia-eye.svg" alt="NVIDIA" width={32} height={21} />
				</div>
				<Breadcrumbs items={items} />
			</div>
			<div className="flex shrink-0 items-center gap-3">
				{rightSlot ? <div className="flex items-center gap-2">{rightSlot}</div> : null}
				<GlobalSearch />
				<UserMenu version={version} />
			</div>
		</header>
	);
};
