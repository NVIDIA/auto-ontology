// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { signOut, useSession } from '@/auth/auth-client';
import { SelectButton } from '@/common/Button';
import { PopoverMenu } from '@/common/PopoverMenu';
import { Text } from '@/common/Text';
import { SelectButtonTheme } from '@/enums/button';
import { TextVariant } from '@/enums/text';

export const UserMenu = ({ version }: { version?: string }) => {
	const { data } = useSession();
	const router = useRouter();
	const [signingOut, setSigningOut] = useState(false);

	if (!data) return null;

	const { user } = data;
	const initial = (user.name || user.email || '?').charAt(0).toUpperCase();
	const fullName = user.name || user.email;
	// Title-case the role, e.g. "admin" → "Admin".
	const roleLabel = user.role
		? user.role.charAt(0).toUpperCase() + user.role.slice(1).toLowerCase()
		: null;

	const handleSignOut = async () => {
		setSigningOut(true);
		await signOut();
		// Hard navigation so the root layout re-runs server-side without a
		// session and drops the app shell (NavRail / top bar).
		window.location.href = '/login';
	};

	return (
		<PopoverMenu
			header={
				// The fixed width is what gives a long email something to be
				// clipped against, since the menu is otherwise as wide as its
				// widest item.
				<div className="w-56 px-3 py-1 text-xs text-zinc-400 dark:text-zinc-500">
					{version ? <p>Version {version}</p> : null}
					<Text text={fullName} variant={TextVariant.Strong} />
					{roleLabel ? <p className="text-[10px]">{roleLabel}</p> : null}
				</div>
			}
			items={[
				{
					// Self-service, so it lives here rather than under the
					// admin-only Settings section.
					label: 'API Tokens',
					onClick: () => router.push('/account/api-tokens'),
				},
				{
					label: signingOut ? 'Signing out…' : 'Sign out',
					onClick: () => {
						void handleSignOut();
					},
					disabled: signingOut,
				},
			]}
			trigger={({ open, toggle }) => (
				<SelectButton
					theme={SelectButtonTheme.Avatar}
					onClick={toggle}
					aria-haspopup="menu"
					aria-expanded={open}
					aria-label="User menu"
				>
					{initial}
				</SelectButton>
			)}
		/>
	);
};
