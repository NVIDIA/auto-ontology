// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import { useSession } from '@/auth/auth-client';
import { Role } from '@/enums/auth';
import { invitationsApi } from '@/api/invitations';
import { usersApi } from '@/api/users';
import { Button, CopyButton } from '@/common/Button';
import { formatDate } from '@/common/date';
import { Icon, IconName } from '@/common/icons';
import { Size, ButtonTheme } from '@/enums/button';
import { InvitationKind, InvitationStatus } from '@/enums/invitation';
import { ToastVariant } from '@/enums/toast';
import { Table } from '@/common/Table';
import { SkeletonTable } from '@/common/Skeleton';
import { Toast } from '@/common/Toast';
import { PopoverMenu } from '@/common/PopoverMenu';
import { ConfirmModal } from '@/common/modal';
import type { TableColumn } from '@/types/table';
import type { UpdateUserInput, User } from '@/types/auth';
import type { Invitation } from '@/types/invitation';
import { EditUserModal } from './EditUserModal';
import { InviteUserModal, type InviteUserInput } from './InviteUserModal';

const invitationStatusLabel: Record<InvitationStatus, string> = {
	[InvitationStatus.Active]: 'Active',
	[InvitationStatus.Expired]: 'Expired',
};

const invitationKindLabel: Record<InvitationKind, string> = {
	[InvitationKind.Invite]: 'Invite',
	[InvitationKind.Reset]: 'Reset password',
};

const actionErrorMessage = (error: unknown, fallback: string): string => {
	if (typeof error === 'object' && error && 'message' in error) {
		const message = (error as { message?: string }).message;
		if (message) return message;
	}
	return fallback;
};

export const UsersManager = () => {
	const { data: session } = useSession();
	const [users, setUsers] = useState<User[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [busyId, setBusyId] = useState<string | null>(null);
	const [inviteOpen, setInviteOpen] = useState(false);
	const [inviting, setInviting] = useState(false);
	const [inviteError, setInviteError] = useState<string | null>(null);
	const [editingUser, setEditingUser] = useState<User | null>(null);
	const [editing, setEditing] = useState(false);
	const [editError, setEditError] = useState<string | null>(null);
	const [pendingDeleteUser, setPendingDeleteUser] = useState<User | null>(null);
	const [invitations, setInvitations] = useState<Invitation[]>([]);

	const currentUserId = session?.user.id;
	const adminCount = users.filter((user) => user.role === Role.Admin).length;
	const editingRoleLockedReason =
		editingUser?.id === currentUserId
			? 'You cannot change your own role.'
			: editingUser?.role === Role.Admin && adminCount <= 1
				? 'The last admin cannot be made a viewer.'
				: null;

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

	const editUser = async (user: User, input: UpdateUserInput) => {
		if (user.id === currentUserId && input.role !== user.role) {
			setEditError('You cannot change your own role.');
			return;
		}
		if (user.role === Role.Admin && adminCount <= 1 && input.role !== Role.Admin) {
			setEditError('The last admin cannot be made a viewer.');
			return;
		}
		setEditing(true);
		setEditError(null);
		const result = await usersApi.update(user.id, input);
		if (result.error) {
			setEditError(actionErrorMessage(result.error, 'Failed to update the user.'));
			setEditing(false);
			return;
		}
		const listed = await usersApi.list();
		if (listed.error) {
			setError(listed.error);
		} else {
			setUsers(listed.users);
		}
		setEditingUser(null);
		setEditing(false);
	};

	const deleteUser = (user: User) => {
		setPendingDeleteUser(user);
	};

	const confirmDeleteUser = async () => {
		if (!pendingDeleteUser) return;
		await runAction(pendingDeleteUser.id, () => usersApi.remove(pendingDeleteUser.id));
		setPendingDeleteUser(null);
	};

	const resetPassword = (user: User) => {
		void (async () => {
			setBusyId(user.id);
			setError(null);
			const result = await invitationsApi.resetPassword(user.id);
			if (result.error) {
				setError(result.message ?? 'Failed to reset the password.');
				setBusyId(null);
				return;
			}
			const [listed, invited] = await Promise.all([usersApi.list(), invitationsApi.list()]);
			setUsers(listed.users);
			if (listed.error) setError(listed.error);
			if (invited.error) setError(invited.message ?? 'Failed to load invitations.');
			else setInvitations(invited.invitations);
			setBusyId(null);
		})();
	};

	const deleteInvitation = (invitation: Invitation) => {
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
			width: 'w-16',
			nowrap: true,
			headerClassName: 'text-right',
			cell: (user) => {
				const isAdmin = user.role === Role.Admin;
				// Never let the last admin be demoted or deleted.
				const isLastAdmin = isAdmin && adminCount <= 1;
				const isSelf = currentUserId != null && user.id === currentUserId;
				const busy = busyId === user.id;
				return (
					<div className="flex justify-end">
						<PopoverMenu
							items={[
								{
									label: 'Edit',
									icon: <Icon name={IconName.Pencil} className="h-3.5 w-3.5" />,
									onClick: () => {
										setEditError(null);
										setEditingUser(user);
									},
								},
								{
									label: 'Reset Password',
									icon: <Icon name={IconName.Key} className="h-3.5 w-3.5" />,
									onClick: () => resetPassword(user),
									disabled: isSelf,
								},
								{
									label: 'Delete',
									icon: <Icon name={IconName.Trash} className="h-3.5 w-3.5" />,
									onClick: () => void deleteUser(user),
									disabled: isLastAdmin,
									danger: true,
								},
							]}
							trigger={({ toggle }) => (
								<Button
									theme={ButtonTheme.IconNeutral}
									size={Size.SMALL}
									iconOnly
									type="button"
									disabled={busy}
									onClick={toggle}
									aria-label={`Actions for ${user.name || user.email}`}
								>
									<Icon name={IconName.DotsVertical} className="h-4 w-4" />
								</Button>
							)}
						/>
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
			key: 'kind',
			header: 'Type',
			width: 'w-36',
			nowrap: true,
			cell: (invitation) => invitationKindLabel[invitation.kind],
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
			<div className="mx-auto max-w-6xl">
				<div className="mb-4">
					<h1 className="mb-1 text-lg font-semibold text-heading dark:text-zinc-100">
						Users
					</h1>
					<p className="text-xs text-secondary">
						Admins manage users; viewers can access all other pages.
					</p>
				</div>

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

				<div className="mt-8">
					<div className="mb-3 flex items-start justify-between gap-4">
						<div className="min-w-0">
							<h2 className="mb-1 text-sm font-semibold text-heading dark:text-zinc-100">
								Invitations
							</h2>
							<p className="text-xs text-secondary">
								Invite by email. Copy the link and send it yourself — they set a
								password on it. A join invite creates the account; a reset invite
								only changes the password.
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
						<div role="status" aria-label="Loading invitations">
							<SkeletonTable columns={5} rows={4} />
						</div>
					) : (
						<Table
							columns={invitationColumns}
							rows={invitations}
							rowKey={(invitation) => invitation.id}
							layout="auto"
							emptyMessage="No Invitations"
						/>
					)}
				</div>
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

			{editingUser ? (
				<EditUserModal
					user={editingUser}
					roleLockedReason={editingRoleLockedReason}
					submitting={editing}
					error={editError}
					onClose={() => {
						if (!editing) setEditingUser(null);
					}}
					onEdit={(input) => {
						void editUser(editingUser, input);
					}}
				/>
			) : null}

			<ConfirmModal
				open={pendingDeleteUser !== null}
				title="Delete User"
				message={
					pendingDeleteUser == null
						? ''
						: `Delete user "${pendingDeleteUser.name || pendingDeleteUser.email}"? This permanently removes the account and its data.`
				}
				confirmLabel="Delete"
				onConfirm={confirmDeleteUser}
				onCancel={() => {
					if (pendingDeleteUser == null || busyId !== pendingDeleteUser.id) {
						setPendingDeleteUser(null);
					}
				}}
				confirming={pendingDeleteUser != null && busyId === pendingDeleteUser.id}
			/>

			<Toast
				open={error !== null}
				message={error ?? ''}
				variant={ToastVariant.Error}
				onClose={() => setError(null)}
			/>
		</div>
	);
};
