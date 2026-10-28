// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Icon, IconName } from '@/common/icons';
import { groupColorClass } from '@/lib/owners';
import type { Owner } from '@/types/owners';

type OwnerLabelOwner = Pick<Owner, 'name' | 'kind' | 'color'>;

const initials = (name: string): string => {
	const parts = name.trim().split(/\s+/).filter(Boolean);
	const letters = parts.slice(0, 2).map((part) => part[0] ?? '');
	return letters.join('').toUpperCase() || '?';
};

/** The owner's name, or "Unassigned" when the id names nobody. */
export const OwnerLabel = ({ owner }: { owner?: OwnerLabelOwner | null }) => {
	if (owner == null) {
		return (
			<span className="inline-flex min-w-0 items-center gap-2 text-sm text-body dark:text-zinc-300">
				<span className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-zinc-200 text-secondary dark:bg-zinc-700 dark:text-zinc-400">
					<Icon name={IconName.User} className="h-3.5 w-3.5" />
				</span>
				<span className="truncate">Unassigned</span>
			</span>
		);
	}
	return (
		<span className="inline-flex min-w-0 items-center gap-2 text-sm text-body dark:text-zinc-300">
			{owner.kind === 'group' ? (
				<span
					className={`inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-white ${groupColorClass(owner.color)}`}
				>
					<Icon name={IconName.Users} className="h-3.5 w-3.5" />
				</span>
			) : (
				<span className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#76b900] text-[10px] font-semibold text-white">
					{initials(owner.name)}
				</span>
			)}
			<span className="truncate">{owner.name}</span>
		</span>
	);
};
