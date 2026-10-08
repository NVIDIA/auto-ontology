// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { invitationsApi } from '@/api/invitations';
import { authApi } from '@/api/auth';
import { Button } from '@/common/Button';
import { Icon, IconName } from '@/common/icons';
import { Size, ButtonTheme } from '@/enums/button';

const inputClass =
	'w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm text-body outline-none transition-colors placeholder:text-secondary focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300';

const MIN_PASSWORD_LENGTH = 8;
const MAX_PASSWORD_LENGTH = 128;

type InviteSetPasswordProps = {
	token: string;
	email: string;
};

export const InviteSetPassword = ({ token, email }: InviteSetPasswordProps) => {
	const router = useRouter();
	const [password, setPassword] = useState('');
	const [confirm, setConfirm] = useState('');
	const [error, setError] = useState<string | null>(null);
	const [submitting, setSubmitting] = useState(false);

	const handleSubmit = async (event: React.FormEvent) => {
		event.preventDefault();
		if (password !== confirm) {
			setError('Passwords do not match.');
			return;
		}
		if (password.length < MIN_PASSWORD_LENGTH) {
			setError(`Password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
			return;
		}
		if (password.length > MAX_PASSWORD_LENGTH) {
			setError(`Password must be at most ${MAX_PASSWORD_LENGTH} characters.`);
			return;
		}

		setSubmitting(true);
		setError(null);
		const accepted = await invitationsApi.accept(token, password);
		if (accepted.error) {
			setError(accepted.message ?? 'Could not accept the invitation.');
			setSubmitting(false);
			return;
		}

		const signedIn = await authApi.signInWithPassword(email, password);
		if (signedIn.error) {
			setError('Account created. Sign in from the login page.');
			setSubmitting(false);
			router.push('/login');
			return;
		}

		router.push('/chat');
		router.refresh();
	};

	return (
		<div className="flex h-full w-full items-center justify-center bg-white p-6 dark:bg-zinc-950">
			<div className="flex w-full max-w-sm flex-col items-center gap-6">
				<div className="flex items-center gap-2">
					<Icon name={IconName.NvidiaLogo} className="h-6 w-6" />
					<span className="text-lg font-semibold text-heading dark:text-zinc-100">
						Auto Ontology
					</span>
				</div>
				<form
					onSubmit={(event) => void handleSubmit(event)}
					className="flex w-full flex-col gap-4"
				>
					<p className="text-xs text-secondary dark:text-zinc-400">
						Set a password for <span className="font-medium text-body">{email}</span> to
						finish joining.
					</p>
					<div className="flex flex-col gap-1.5">
						<label
							htmlFor="invite-email"
							className="text-xs font-medium text-body dark:text-zinc-400"
						>
							Email
						</label>
						<input
							id="invite-email"
							type="email"
							readOnly
							value={email}
							className={`${inputClass} cursor-not-allowed bg-zinc-50 dark:bg-zinc-800/60`}
						/>
					</div>
					<div className="flex flex-col gap-1.5">
						<label
							htmlFor="invite-password"
							className="text-xs font-medium text-body dark:text-zinc-400"
						>
							Password
						</label>
						<input
							id="invite-password"
							type="password"
							required
							minLength={MIN_PASSWORD_LENGTH}
							maxLength={MAX_PASSWORD_LENGTH}
							autoComplete="new-password"
							value={password}
							onChange={(event) => setPassword(event.target.value)}
							className={inputClass}
						/>
					</div>
					<div className="flex flex-col gap-1.5">
						<label
							htmlFor="invite-password-confirm"
							className="text-xs font-medium text-body dark:text-zinc-400"
						>
							Confirm password
						</label>
						<input
							id="invite-password-confirm"
							type="password"
							required
							minLength={MIN_PASSWORD_LENGTH}
							maxLength={MAX_PASSWORD_LENGTH}
							autoComplete="new-password"
							value={confirm}
							onChange={(event) => setConfirm(event.target.value)}
							className={inputClass}
						/>
					</div>
					{error ? <p className="text-xs text-red-500">{error}</p> : null}
					<Button
						theme={ButtonTheme.Primary}
						size={Size.REGULAR}
						type="submit"
						disabled={submitting}
					>
						{submitting ? 'Saving…' : 'Create password'}
					</Button>
				</form>
			</div>
		</div>
	);
};
