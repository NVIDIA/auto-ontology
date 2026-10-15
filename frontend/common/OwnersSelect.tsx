// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Popover } from '@/common/Popover';
import { PopoverAlign } from '@/enums/popover';
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

/**
 * Several users and groups, opened from a field that matches {@link OwnersSelect}.
 * The menu stays open so more than one row can be chosen.
 */
export const OwnersMultiSelect = ({
	owners,
	value,
	onChange,
	disabled = false,
	emptyText = 'No users or groups available.',
}: {
	owners: OwnerOption[];
	value: string[];
	onChange: (ids: string[]) => void;
	disabled?: boolean;
	emptyText?: string;
}) => {
	const toggleId = (id: string) => {
		onChange(value.includes(id) ? value.filter((item) => item !== id) : [...value, id]);
	};
	const users = owners.filter((owner) => owner.kind === 'user');
	const groups = owners.filter((owner) => owner.kind === 'group');
	const selected = owners.filter((owner) => value.includes(owner.id));
	const label =
		selected.length > 0 ? selected.map((owner) => owner.name).join(', ') : 'Unassigned';
	const unavailable = owners.length === 0;

	const renderGroup = (title: string, list: OwnerOption[]) =>
		list.length === 0 ? null : (
			<div>
				<p className="px-2 py-1 text-xs font-medium text-secondary">{title}</p>
				<ul>
					{list.map((owner) => {
						const checked = value.includes(owner.id);
						return (
							<li key={owner.id}>
								<button
									type="button"
									className={`flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm text-heading hover:bg-zinc-50 dark:text-zinc-100 dark:hover:bg-zinc-700/60 ${checked ? 'bg-zinc-50 dark:bg-zinc-700/40' : ''}`}
									aria-pressed={checked}
									onClick={() => toggleId(owner.id)}
								>
									<span className="min-w-0 flex-1 truncate">{owner.name}</span>
									{checked ? (
										<span className="text-[#76b900]" aria-hidden>
											✓
										</span>
									) : null}
								</button>
							</li>
						);
					})}
				</ul>
			</div>
		);

	return (
		<Popover
			align={PopoverAlign.Stretch}
			className="w-full"
			panelClassName="max-h-60 overflow-y-auto p-1"
			trigger={({ open, toggle }) => (
				<button
					type="button"
					className={`${FIELD_CLASS} flex items-center justify-between gap-2 text-left disabled:cursor-not-allowed disabled:opacity-60`}
					disabled={disabled || unavailable}
					aria-label="Zone members"
					aria-expanded={open}
					aria-haspopup="listbox"
					onClick={(event) => {
						if (disabled || unavailable) return;
						toggle(event);
					}}
				>
					<span
						className={`min-w-0 flex-1 truncate ${selected.length === 0 ? 'text-secondary' : ''}`}
					>
						{unavailable ? emptyText : label}
					</span>
					<svg
						className="h-4 w-4 shrink-0 text-secondary"
						viewBox="0 0 20 20"
						fill="currentColor"
						aria-hidden
					>
						<path
							fillRule="evenodd"
							d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.17l3.71-3.94a.75.75 0 1 1 1.08 1.04l-4.25 4.5a.75.75 0 0 1-1.08 0l-4.25-4.5a.75.75 0 0 1 .02-1.06Z"
							clipRule="evenodd"
						/>
					</svg>
				</button>
			)}
		>
			{() => (
				<div role="listbox" aria-label="Zone members" aria-multiselectable>
					{renderGroup('Users', users)}
					{renderGroup('Groups', groups)}
				</div>
			)}
		</Popover>
	);
};
