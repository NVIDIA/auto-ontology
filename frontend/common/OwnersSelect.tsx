// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { OwnerKind } from '@/types/owners';

export type OwnerOption = {
	id: string;
	name: string;
	kind: OwnerKind;
	color: string | null;
};

const FIELD_CLASS =
	'w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-body outline-none transition-colors focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300';

/**
 * One owner, or nobody. Users and groups are separate groups in the list,
 * which is how an id that could be either stays obvious.
 */
export const OwnersSelect = ({
	owners,
	value,
	onChange,
	disabled = false,
}: {
	owners: OwnerOption[];
	/** Empty when the object is unassigned. */
	value: string;
	onChange: (ownerId: string) => void;
	disabled?: boolean;
}) => {
	const users = owners.filter((owner) => owner.kind === 'user');
	const groups = owners.filter((owner) => owner.kind === 'group');
	return (
		<select
			className={FIELD_CLASS}
			value={value}
			disabled={disabled}
			onChange={(event) => onChange(event.target.value)}
			aria-label="Owner"
		>
			<option value="">Unassigned</option>
			{users.length > 0 && (
				<optgroup label="Users">
					{users.map((owner) => (
						<option key={owner.id} value={owner.id}>
							{owner.name}
						</option>
					))}
				</optgroup>
			)}
			{groups.length > 0 && (
				<optgroup label="Groups">
					{groups.map((owner) => (
						<option key={owner.id} value={owner.id}>
							{owner.name}
						</option>
					))}
				</optgroup>
			)}
		</select>
	);
};
