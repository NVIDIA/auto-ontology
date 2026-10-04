// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import { Role } from '@/enums/auth';
import { usersApi } from '@/api/users';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import { ToastVariant } from '@/enums/toast';
import { Table } from '@/common/Table';
import { SkeletonTable } from '@/common/Skeleton';
import { Toast } from '@/common/Toast';
import type { TableColumn } from '@/types/table';
import type { User } from '@/types/auth';

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
					<div className="text-xs text-secondary">{user.email}</div>
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
						<Button
							theme={ButtonTheme.Secondary}
							size={Size.SMALL}
							type="button"
							disabled={busy || isLastAdmin}
							onClick={() => toggleRole(user)}
						>
							{isAdmin ? 'Make viewer' : 'Make admin'}
						</Button>
						<Button
							theme={ButtonTheme.DangerOutline}
							size={Size.SMALL}
							type="button"
							disabled={busy || isLastAdmin}
							onClick={() => deleteUser(user)}
						>
							Delete
						</Button>
					</div>
				);
			},
		},
	];

	return (
		<div className="h-full overflow-auto p-6">
			<div className="mx-auto max-w-3xl">
				<h1 className="mb-1 text-lg font-semibold text-heading dark:text-zinc-100">
					Users
				</h1>
				<p className="mb-4 text-xs text-secondary">
					Manage roles and access. Admins manage users; viewers can access all other
					pages.
				</p>

				{loading ? (
					<div role="status" aria-label="Loading users">
						<SkeletonTable columns={3} rows={8} />
					</div>
				) : (
					<Table
						columns={columns}
						rows={users}
						rowKey={(user) => user.id}
						layout="auto"
						emptyMessage="No Users Found"
					/>
				)}
			</div>

			<Toast
				open={error !== null}
				message={error ?? ''}
				variant={ToastVariant.Error}
				onClose={() => setError(null)}
			/>
		</div>
	);
};
