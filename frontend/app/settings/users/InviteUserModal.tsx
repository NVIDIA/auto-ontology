// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';
import { Button } from '@/common/Button';
import { Modal } from '@/common/modal';
import { Size, ButtonTheme } from '@/enums/button';
import { Role } from '@/enums/auth';

const inputClass =
	'w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-heading outline-none focus:border-[#76b900] focus:ring-1 focus:ring-[#76b900] dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100';

export type InviteUserInput = {
	email: string;
	name: string;
	role: Role;
};

type InviteUserModalProps = {
	submitting: boolean;
	error: string | null;
	onClose: () => void;
	onInvite: (input: InviteUserInput) => void;
};

export const InviteUserModal = ({ submitting, error, onClose, onInvite }: InviteUserModalProps) => {
	const [email, setEmail] = useState('');
	const [name, setName] = useState('');
	const [role, setRole] = useState<Role>(Role.Viewer);

	return (
		<Modal open onClose={onClose} className="w-full max-w-md">
			<form
				className="space-y-4 p-5"
				onSubmit={(event) => {
					event.preventDefault();
					onInvite({ email, name, role });
				}}
			>
				<h2 className="text-sm font-semibold text-heading dark:text-zinc-100">
					Invite user
				</h2>
				<p className="text-xs text-secondary dark:text-zinc-400">
					Creates a one-time link. Copy it and send it yourself — no email is sent. They
					set their own password on the link.
				</p>

				<label className="block space-y-1.5" htmlFor="invite-email">
					<span className="text-xs font-medium text-body dark:text-zinc-300">Email</span>
					<input
						id="invite-email"
						className={inputClass}
						type="email"
						required
						autoComplete="off"
						value={email}
						onChange={(event) => setEmail(event.target.value)}
					/>
				</label>

				<label className="block space-y-1.5" htmlFor="invite-name">
					<span className="text-xs font-medium text-body dark:text-zinc-300">
						Name <span className="font-normal text-secondary">(optional)</span>
					</span>
					<input
						id="invite-name"
						className={inputClass}
						type="text"
						autoComplete="off"
						value={name}
						onChange={(event) => setName(event.target.value)}
					/>
				</label>

				<label className="block space-y-1.5" htmlFor="invite-role">
					<span className="text-xs font-medium text-body dark:text-zinc-300">Role</span>
					<select
						id="invite-role"
						className={inputClass}
						value={role}
						onChange={(event) =>
							setRole(event.target.value === Role.Admin ? Role.Admin : Role.Viewer)
						}
					>
						<option value={Role.Viewer}>Viewer</option>
						<option value={Role.Admin}>Admin</option>
					</select>
				</label>

				{error ? <p className="text-xs text-red-600 dark:text-red-400">{error}</p> : null}

				<div className="flex justify-end gap-2">
					<Button
						theme={ButtonTheme.Secondary}
						size={Size.SMALL}
						type="button"
						onClick={onClose}
					>
						Cancel
					</Button>
					<Button
						theme={ButtonTheme.Primary}
						size={Size.SMALL}
						type="submit"
						disabled={submitting}
					>
						{submitting ? 'Inviting…' : 'Invite'}
					</Button>
				</div>
			</form>
		</Modal>
	);
};
