// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';
import { signOut, useSession } from '@/auth/auth-client';

export const UserMenu = () => {
	const { data } = useSession();
	const [signingOut, setSigningOut] = useState(false);

	if (!data) return null;

	const { user } = data;
	const initial = (user.name || user.email || '?').charAt(0).toUpperCase();
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
		<div className="flex items-center gap-3">
			<div className="flex items-center gap-2">
				<div className="flex h-7 w-7 items-center justify-center rounded-full bg-[#76b900]/15 text-xs font-semibold text-[#76b900]">
					{initial}
				</div>
				<div className="hidden flex-col leading-tight sm:flex">
					<span className="text-xs font-medium text-zinc-700 dark:text-zinc-200">
						{user.name || user.email}
					</span>
					<span className="text-[10px] tracking-wide text-zinc-400">{roleLabel}</span>
				</div>
			</div>
			<button
				type="button"
				onClick={handleSignOut}
				disabled={signingOut}
				className="cursor-pointer rounded-md border border-zinc-300 px-2.5 py-1 text-xs font-medium text-zinc-600 transition-colors hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-600 dark:text-zinc-400 dark:hover:bg-zinc-800"
			>
				{signingOut ? 'Signing out…' : 'Sign out'}
			</button>
		</div>
	);
};
