// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import { Role } from '@/enums/auth';
import { invitationsApi } from '@/api/invitations';
import { usersApi } from '@/api/users';
import { Button, CopyButton } from '@/common/Button';
import { formatDate } from '@/common/date';
import { Icon, IconName } from '@/common/icons';
import { Size, ButtonTheme } from '@/enums/button';
import { InvitationStatus } from '@/enums/invitation';
import { ToastVariant } from '@/enums/toast';
import { Table } from '@/common/Table';
import { SkeletonTable } from '@/common/Skeleton';
import { Toast } from '@/common/Toast';
import type { TableColumn } from '@/types/table';
import type { User } from '@/types/auth';
import type { Invitation } from '@/types/invitation';
import { InviteUserModal, type InviteUserInput } from './InviteUserModal';

const invitationStatusLabel: Record<InvitationStatus, string> = {
	[InvitationStatus.Active]: 'Active',
	[InvitationStatus.Expired]: 'Expired',
};

const actionErrorMessage = (error: unknown, fallback: string): string => {
	if (typeof error === 'object' && error && 'message' in error) {
		const message = (error as { message?: string }).message;
		if (message) return message;
	}
	return fallback;
};

export const UsersManager = () => {
	const [users, setUsers] = useState<User[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [busyId, setBusyId] = useState<string | null>(null);
	const [inviteOpen, setInviteOpen] = useState(false);
	const [inviting, setInviting] = useState(false);
	const [inviteError, setInviteError] = useState<string | null>(null);
	const [invitations, setInvitations] = useState<Invitation[]>([]);

	const adminCount = users.filter((user) => user.role === Role.Admin).length;

	useEffect(() => {
		void Promise.all([usersApi.list(), invitationsApi.list()]).then(([listed, invited]) => {
			setUsers(listed.users);
			setError(listed.error ?? (invited.error ? (invited.message ?? null) : null));
			if (!invited.error) setInvitations(invited.invitations);
			setLoading(false);
		});
	}, []);

	const runAction = async (id: string, action: () => Promise<{ error?: unknown }>) => {
		setBusyId(id);
		setError(null);
		const actionResult = await action();
		if (actionResult.error) {
			setError(actionErrorMessage(actionResult.error, 'Action failed.'));
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

	const inviteUser = async ({ email, name, role }: InviteUserInput) => {
		setInviting(true);
		setInviteError(null);
		const result = await invitationsApi.create({ email, name, role });
		if (result.error) {
			setInviteError(result.message ?? 'Failed to invite the user.');
			setInviting(false);
			return;
		}
		setInviteOpen(false);
		setInviting(false);
		const [listed, invited] = await Promise.all([usersApi.list(), invitationsApi.list()]);
		setUsers(listed.users);
		if (listed.error) setError(listed.error);
		if (invited.error) setError(invited.message ?? 'Failed to load invitations.');
		else setInvitations(invited.invitations);
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

	const deleteInvitation = (invitation: Invitation) => {
		if (
			!window.confirm(
				`Delete invitation for "${invitation.email}"? If the link is still unused, it will stop working.`,
			)
		) {
			return;
		}
		void (async () => {
			setBusyId(invitation.id);
			setError(null);
			const result = await invitationsApi.remove(invitation.id);
			if (result.error) {
				setError(result.message ?? 'Failed to delete the invitation.');
				setBusyId(null);
				return;
			}
			setInvitations((rows) => rows.filter((row) => row.id !== invitation.id));
			setBusyId(null);
		})();
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

	const invitationColumns: TableColumn<Invitation>[] = [
		{
			key: 'email',
			header: 'Email',
			cell: (invitation) => invitation.email,
		},
		{
			key: 'role',
			header: 'Role',
			width: 'w-28',
			nowrap: true,
			cell: (invitation) => (invitation.role === Role.Admin ? 'Admin' : 'Viewer'),
		},
		{
			key: 'status',
			header: 'Status',
			width: 'w-28',
			nowrap: true,
			cell: (invitation) => invitationStatusLabel[invitation.status],
		},
		{
			key: 'expires',
			header: 'Expires',
			width: 'w-40',
			nowrap: true,
			cell: (invitation) => formatDate(invitation.expires_at, 'MMM DD YYYY HH:mm'),
		},
		{
			key: 'link',
			header: 'Link',
			truncate: true,
			maxWidthClass: 'max-w-md',
			title: (invitation) => invitation.url ?? '',
			cell: (invitation) =>
				invitation.url ? (
					<div className="group flex min-w-0 items-center gap-1">
						<code
							className="min-w-0 truncate font-mono text-xs text-body dark:text-zinc-300"
							title={invitation.url}
						>
							{invitation.url}
						</code>
						<CopyButton text={invitation.url} className="shrink-0 opacity-100" />
					</div>
				) : (
					<span className="text-secondary">—</span>
				),
		},
		{
			key: 'actions',
			header: 'Actions',
			width: 'w-28',
			nowrap: true,
			headerClassName: 'text-right',
			cell: (invitation) => (
				<div className="flex justify-end">
					<Button
						theme={ButtonTheme.DangerOutline}
						size={Size.SMALL}
						type="button"
						disabled={busyId === invitation.id}
						onClick={() => deleteInvitation(invitation)}
					>
						Delete
					</Button>
				</div>
			),
		},
	];

	return (
		<div className="h-full overflow-auto p-6">
			<div className="mx-auto max-w-5xl">
				<div className="mb-4 flex items-start justify-between gap-4">
					<div className="min-w-0">
						<h1 className="mb-1 text-lg font-semibold text-heading dark:text-zinc-100">
							Users
						</h1>
						<p className="text-xs text-secondary">
							Invite by email. Copy the link and send it yourself — they set a
							password on it. Admins manage users; viewers can access all other pages.
						</p>
					</div>
					<Button
						theme={ButtonTheme.Primary}
						size={Size.SMALL}
						type="button"
						onClick={() => {
							setInviteError(null);
							setInviteOpen(true);
						}}
					>
						<Icon name={IconName.Plus} className="mr-1.5 h-4 w-4" />
						Invite
					</Button>
				</div>

				{loading ? (
					<div role="status" aria-label="Loading users">
						<SkeletonTable columns={3} rows={8} />
					</div>
				) : (
					<div className="space-y-8">
						<Table
							columns={columns}
							rows={users}
							rowKey={(user) => user.id}
							layout="auto"
							emptyMessage="No Users Found"
						/>
						<div>
							<h2 className="mb-1 text-sm font-semibold text-heading dark:text-zinc-100">
								Invitations
							</h2>
							<p className="mb-3 text-xs text-secondary">
								Active links can be copied again. Accepted invites are removed once
								the account is created. Delete a row to revoke an unused link or
								clear an expired invite.
							</p>
							<Table
								columns={invitationColumns}
								rows={invitations}
								rowKey={(invitation) => invitation.id}
								layout="auto"
								emptyMessage="No Invitations"
							/>
						</div>
					</div>
				)}
			</div>

			{inviteOpen ? (
				<InviteUserModal
					submitting={inviting}
					error={inviteError}
					onClose={() => setInviteOpen(false)}
					onInvite={(input) => {
						void inviteUser(input);
					}}
				/>
			) : null}

			<Toast
				open={error !== null}
				message={error ?? ''}
				variant={ToastVariant.Error}
				onClose={() => setError(null)}
			/>
		</div>
	);
};
