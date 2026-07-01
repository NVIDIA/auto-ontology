// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import { Spinner } from '@nvidia/foundations-react-core';
import { Role } from '@/enums/auth';
import { usersApi } from '@/api/users';
import { Table } from '@/components/Table';
import { Toast } from '@/components/Toast';
import type { TableColumn } from '@/types/table';
import type { User } from '@/types/auth';

const actionButtonClass =
	'cursor-pointer rounded-md border border-zinc-300 px-2 py-1 text-xs font-medium text-zinc-600 transition-colors hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-600 dark:text-zinc-400 dark:hover:bg-zinc-800';

const deleteButtonClass =
	'cursor-pointer rounded-md border border-red-300 px-2 py-1 text-xs font-medium text-red-600 transition-colors hover:bg-red-50 disabled:opacity-40 dark:border-red-800 dark:text-red-400 dark:hover:bg-red-950';

export const UsersManager = () => {
	const [users, setUsers] = useState<User[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [busyId, setBusyId] = useState<string | null>(null);

	const adminCount = users.filter((user) => user.role === Role.Admin).length;

	useEffect(() => {
		usersApi.list().then((result) => {
			setUsers(result.users);
			setError(result.error);
			setLoading(false);
		});
	}, []);

	const runAction = async (id: string, action: () => Promise<{ error?: unknown }>) => {
		setBusyId(id);
		setError(null);
		const actionResult = await action();
		if (actionResult.error) {
			const message =
				typeof actionResult.error === 'object' &&
				actionResult.error &&
				'message' in actionResult.error
					? String((actionResult.error as { message?: string }).message)
					: 'Action failed.';
			setError(message);
		}
		const listed = await usersApi.list();
		setUsers(listed.users);
		if (listed.error) setError(listed.error);
		setBusyId(null);
	};

	const toggleRole = (user: User) => {
		const nextRole = user.role === Role.Admin ? Role.Viewer : Role.Admin;
		return runAction(user.id, () => usersApi.setRole(user.id, nextRole));
	};

	const deleteUser = (user: User) => {
		if (
			!window.confirm(
				`Delete user "${user.name || user.email}"? This permanently removes the account and its data.`,
			)
		) {
			return;
		}
		return runAction(user.id, () => usersApi.remove(user.id));
	};

	const columns: TableColumn<User>[] = [
		{
			key: 'user',
			header: 'User',
			cell: (user) => (
				<div>
					<div className="font-medium">{user.name || '—'}</div>
					<div className="text-xs text-zinc-400">{user.email}</div>
				</div>
			),
		},
		{
			key: 'role',
			header: 'Role',
			width: 'w-32',
			nowrap: true,
			cell: (user) => (user.role === Role.Admin ? 'Admin' : 'Viewer'),
		},
		{
			key: 'actions',
			header: 'Actions',
			width: 'w-56',
			nowrap: true,
			headerClassName: 'text-right',
			cell: (user) => {
				const isAdmin = user.role === Role.Admin;
				// Never let the last admin be demoted or deleted.
				const isLastAdmin = isAdmin && adminCount <= 1;
				const busy = busyId === user.id;
				return (
					<div className="flex justify-end gap-2">
						<button
							type="button"
							disabled={busy || isLastAdmin}
							onClick={() => toggleRole(user)}
							className={actionButtonClass}
						>
							{isAdmin ? 'Make viewer' : 'Make admin'}
						</button>
						<button
							type="button"
							disabled={busy || isLastAdmin}
							onClick={() => deleteUser(user)}
							className={deleteButtonClass}
						>
							Delete
						</button>
					</div>
				);
			},
		},
	];

	return (
		<div className="h-full overflow-auto p-6">
			<div className="mx-auto max-w-3xl">
				<h1 className="mb-1 text-lg font-semibold text-zinc-900 dark:text-zinc-100">
					Users
				</h1>
				<p className="mb-4 text-xs text-zinc-500">
					Manage roles and access. Admins manage users; viewers can access all other
					pages.
				</p>

				{loading ? (
					<Spinner aria-label="Loading users" />
				) : (
					<Table
						columns={columns}
						rows={users}
						rowKey={(user) => user.id}
						layout="auto"
					/>
				)}
			</div>

			<Toast
				open={error !== null}
				message={error ?? ''}
				variant="error"
				onClose={() => setError(null)}
			/>
		</div>
	);
};
