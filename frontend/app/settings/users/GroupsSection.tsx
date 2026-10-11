// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { groupsApi } from '@/api/groups';
import { Button } from '@/common/Button';
import { ConfirmModal, ModalCreateNewItem } from '@/common/modal';
import { SkeletonTable } from '@/common/Skeleton';
import { Table } from '@/common/Table';
import { ColorPicker } from '@/components/settings/ColorPicker';
import type { ColorOption } from '@/components/settings/ColorPicker';
import { Size, ButtonTheme } from '@/enums/button';
import { Role } from '@/enums/auth';
import type { User } from '@/types/auth';
import { MAX_GROUP_NAME_LENGTH, type Group } from '@/types/groups';
import type { TableColumn } from '@/types/table';

const GROUP_COLORS: readonly ColorOption[] = [
	{ value: '#0ea5e9', label: 'Sky', colorClass: 'bg-sky-500' },
	{ value: '#8b5cf6', label: 'Violet', colorClass: 'bg-violet-500' },
	{ value: '#ec4899', label: 'Pink', colorClass: 'bg-pink-500' },
	{ value: '#f97316', label: 'Orange', colorClass: 'bg-orange-500' },
	{ value: '#14b8a6', label: 'Teal', colorClass: 'bg-teal-500' },
	{ value: '#eab308', label: 'Amber', colorClass: 'bg-yellow-500' },
];

const DEFAULT_GROUP_COLOR = GROUP_COLORS[0].value;

const inputClassName =
	'w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-heading outline-none focus:border-[#76b900] focus:ring-1 focus:ring-[#76b900] dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100';

const labelClassName = 'mb-1.5 block text-sm font-semibold text-heading dark:text-zinc-100';

type Draft = {
	id: string | null;
	name: string;
	description: string;
	color: string;
	role: Role;
	userIds: string[];
};

const emptyDraft = (): Draft => ({
	id: null,
	name: '',
	description: '',
	color: DEFAULT_GROUP_COLOR,
	role: Role.Viewer,
	userIds: [],
});

const colorClassFor = (value: string): string =>
	GROUP_COLORS.find((color) => color.value === value)?.colorClass ?? 'bg-zinc-400';

const asRole = (role: string): Role => (role === Role.Admin ? Role.Admin : Role.Viewer);

const memberNames = (group: Group): string => {
	const names = group.users.map((user) => user.name || user.email || user.id);
	if (names.length === 0) return '—';
	if (names.length <= 3) return names.join(', ');
	return `${names.slice(0, 3).join(', ')} +${names.length - 3}`;
};

const messageOf = (result: { error?: boolean; message?: string }, fallback: string): string =>
	result.message || fallback;

type GroupsSectionProps = {
	/** Accounts already loaded for the users table, offered as members. */
	users: User[];
	/** True while that account list is still loading. Saving waits for it. */
	usersLoading: boolean;
	/** Bumped after an account is deleted so membership is read again. */
	refreshToken: number;
};

export const GroupsSection = ({ users, usersLoading, refreshToken }: GroupsSectionProps) => {
	const [groups, setGroups] = useState<Group[]>([]);
	const [loading, setLoading] = useState(true);
	const [loadError, setLoadError] = useState<string | null>(null);
	const [draft, setDraft] = useState<Draft | null>(null);
	const [saving, setSaving] = useState(false);
	const [submitError, setSubmitError] = useState<string | null>(null);
	const [pendingDelete, setPendingDelete] = useState<Group | null>(null);
	const [deleting, setDeleting] = useState(false);
	const [deleteError, setDeleteError] = useState<string | null>(null);

	const load = useCallback(async (isCurrent: () => boolean = () => true) => {
		const result = await groupsApi.getAll();
		if (!isCurrent()) return;
		if (result.error) {
			setLoadError(messageOf(result, 'Failed to load groups.'));
			setGroups([]);
		} else {
			setLoadError(null);
			setGroups(result.data ?? []);
		}
		setLoading(false);
	}, []);

	useEffect(() => {
		let cancelled = false;
		// `load` writes state only after the request resolves. The rule sees the
		// call itself, which returns a promise, as a synchronous setState.
		// eslint-disable-next-line react-hooks/set-state-in-effect -- awaited inside load
		void load(() => !cancelled);
		return () => {
			cancelled = true;
		};
	}, [load, refreshToken]);

	const sortedUsers = useMemo(
		() =>
			[...users].sort((left, right) =>
				(left.name || left.email).localeCompare(right.name || right.email),
			),
		[users],
	);

	const openCreate = () => {
		setSubmitError(null);
		setDraft(emptyDraft());
	};

	const openEdit = (group: Group) => {
		setSubmitError(null);
		setDraft({
			id: group.id,
			name: group.name,
			description: group.description,
			color: group.color,
			role: asRole(group.role),
			userIds: group.users.map((user) => user.id),
		});
	};

	const closeDraft = () => {
		if (saving) return;
		setDraft(null);
		setSubmitError(null);
	};

	const toggleMember = (userId: string) => {
		setDraft((current) => {
			if (current == null) return current;
			const userIds = current.userIds.includes(userId)
				? current.userIds.filter((id) => id !== userId)
				: [...current.userIds, userId];
			return { ...current, userIds };
		});
	};

	const save = async () => {
		if (draft == null) return;
		const name = draft.name.trim();
		if (name === '') return;
		setSaving(true);
		setSubmitError(null);
		const known = new Set(users.map((user) => user.id));
		const existing =
			draft.id == null ? undefined : groups.find((group) => group.id === draft.id);
		// A deleted account stays on the group with an empty email. Drop only
		// those ids: the backend rejects them. Keep every other id, including
		// one the account list did not return, so a failed or partial user
		// load cannot clear the membership.
		const stillMembers = new Set(
			(existing?.users ?? [])
				.filter((member) => member.email !== '')
				.map((member) => member.id),
		);
		const input = {
			name,
			description: draft.description.trim() === '' ? null : draft.description.trim(),
			color: draft.color,
			role: draft.role,
			user_ids: draft.userIds.filter((id) => known.has(id) || stillMembers.has(id)),
		};
		const result =
			draft.id == null
				? await groupsApi.create(input)
				: await groupsApi.update(draft.id, input);
		setSaving(false);
		if (result.error || result.data == null) {
			setSubmitError(messageOf(result, 'Could not save the group.'));
			return;
		}
		setDraft(null);
		await load();
	};

	const confirmDelete = async () => {
		if (pendingDelete == null) return;
		setDeleting(true);
		setDeleteError(null);
		const result = await groupsApi.delete(pendingDelete.id);
		setDeleting(false);
		if (result.error) {
			setDeleteError(messageOf(result, 'Could not delete the group.'));
			return;
		}
		setPendingDelete(null);
		await load();
	};

	const columns: TableColumn<Group>[] = [
		{
			key: 'name',
			header: 'Group',
			cell: (group) => (
				<div className="flex items-center gap-2">
					<span
						className={`inline-block h-3 w-3 shrink-0 rounded-full ${colorClassFor(group.color)}`}
						title={group.color}
					/>
					<div className="min-w-0">
						<div className="font-medium">{group.name}</div>
						{group.description ? (
							<div className="truncate text-xs text-secondary">
								{group.description}
							</div>
						) : null}
					</div>
				</div>
			),
		},
		{
			key: 'role',
			header: 'Role',
			width: 'w-28',
			nowrap: true,
			cell: (group) => (group.role === Role.Admin ? 'Admin' : 'Viewer'),
		},
		{
			key: 'members',
			header: 'Members',
			cell: (group) => (
				<span className="text-secondary" title={memberNames(group)}>
					{memberNames(group)}
				</span>
			),
		},
		{
			key: 'actions',
			header: 'Actions',
			width: 'w-40',
			nowrap: true,
			headerClassName: 'text-right',
			cell: (group) => (
				<div className="flex justify-end gap-2">
					<Button
						theme={ButtonTheme.Secondary}
						size={Size.SMALL}
						type="button"
						onClick={() => openEdit(group)}
					>
						Edit
					</Button>
					<Button
						theme={ButtonTheme.DangerOutline}
						size={Size.SMALL}
						type="button"
						onClick={() => {
							setDeleteError(null);
							setPendingDelete(group);
						}}
					>
						Delete
					</Button>
				</div>
			),
		},
	];

	const canSave = (draft?.name.trim() ?? '') !== '' && !saving && !usersLoading;

	return (
		<section className="mt-10">
			<div className="mb-4 flex items-start justify-between gap-4">
				<div>
					<h2 className="text-lg font-semibold text-heading dark:text-zinc-100">
						Groups
					</h2>
					<p className="mt-1 text-xs text-secondary">
						A group has its own role. Members keep the role on their account.
					</p>
				</div>
				<Button
					theme={ButtonTheme.Primary}
					size={Size.SMALL}
					type="button"
					onClick={openCreate}
				>
					Create group
				</Button>
			</div>

			{loadError ? (
				<p className="mb-3 text-xs text-red-700 dark:text-red-300" role="alert">
					{loadError}
				</p>
			) : null}

			{loading ? (
				<div role="status" aria-label="Loading groups">
					<SkeletonTable columns={4} rows={4} />
				</div>
			) : (
				<Table
					columns={columns}
					rows={groups}
					rowKey={(group) => group.id}
					layout="auto"
					emptyMessage="No Groups Found"
				/>
			)}

			<ModalCreateNewItem
				open={draft != null}
				onClose={closeDraft}
				title={draft?.id == null ? 'Create Group' : 'Edit Group'}
				submitLabel={draft?.id == null ? 'Create' : 'Save'}
				canSubmit={canSave}
				onSubmit={save}
				className="w-[560px] max-w-full"
			>
				{draft ? (
					<>
						<div>
							<label htmlFor="group-name" className={labelClassName}>
								Name
								<span className="ml-2 font-normal text-secondary dark:text-zinc-500">
									(Max. {MAX_GROUP_NAME_LENGTH})
								</span>
							</label>
							<input
								id="group-name"
								type="text"
								value={draft.name}
								maxLength={MAX_GROUP_NAME_LENGTH}
								onChange={(event) => {
									setDraft({ ...draft, name: event.target.value });
									setSubmitError(null);
								}}
								placeholder="Finance"
								className={inputClassName}
							/>
						</div>
						<div>
							<label htmlFor="group-description" className={labelClassName}>
								Description
							</label>
							<textarea
								id="group-description"
								value={draft.description}
								rows={2}
								onChange={(event) =>
									setDraft({ ...draft, description: event.target.value })
								}
								className={inputClassName}
							/>
						</div>
						<div className="flex items-end gap-4">
							<div>
								<span className={labelClassName}>Color</span>
								<ColorPicker
									colors={GROUP_COLORS}
									value={draft.color}
									onChange={(color) => setDraft({ ...draft, color })}
								/>
							</div>
							<div className="min-w-0 flex-1">
								<label htmlFor="group-role" className={labelClassName}>
									Role
								</label>
								<select
									id="group-role"
									className={inputClassName}
									value={draft.role}
									onChange={(event) =>
										setDraft({ ...draft, role: asRole(event.target.value) })
									}
								>
									<option value={Role.Viewer}>Viewer</option>
									<option value={Role.Admin}>Admin</option>
								</select>
							</div>
						</div>
						<div>
							<span className={labelClassName}>Members</span>
							{usersLoading ? (
								<p className="text-xs text-secondary">Loading users…</p>
							) : sortedUsers.length === 0 ? (
								<p className="text-xs text-secondary">No users to add.</p>
							) : (
								<ul className="max-h-48 space-y-1 overflow-y-auto rounded-lg border border-zinc-200 p-2 dark:border-zinc-700">
									{sortedUsers.map((user) => {
										const checked = draft.userIds.includes(user.id);
										const label = user.name || user.email;
										const showEmail =
											Boolean(user.name) && user.name !== user.email;
										return (
											<li key={user.id}>
												<label className="flex cursor-pointer items-center gap-2 rounded px-1 py-1 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-800/60">
													<input
														type="checkbox"
														checked={checked}
														onChange={() => toggleMember(user.id)}
													/>
													<span className="min-w-0 truncate text-heading dark:text-zinc-100">
														{label}
													</span>
													{showEmail ? (
														<span className="truncate text-xs text-secondary">
															{user.email}
														</span>
													) : null}
												</label>
											</li>
										);
									})}
								</ul>
							)}
						</div>
						{submitError ? (
							<div
								className="rounded-lg border border-red-200/90 bg-red-50 px-3 py-2 text-xs text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200"
								role="alert"
							>
								{submitError}
							</div>
						) : null}
						<div className="-mx-6 border-t border-zinc-200 dark:border-zinc-700" />
					</>
				) : null}
			</ModalCreateNewItem>

			<ConfirmModal
				open={pendingDelete != null}
				title="Delete Group"
				message={
					pendingDelete == null
						? ''
						: `Delete group "${pendingDelete.name}"? Members keep their accounts.`
				}
				onConfirm={confirmDelete}
				onCancel={() => {
					if (deleting) return;
					setPendingDelete(null);
					setDeleteError(null);
				}}
				confirming={deleting}
				error={deleteError}
			/>
		</section>
	);
};
