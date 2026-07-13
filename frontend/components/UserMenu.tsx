// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef, useState } from 'react';
import { signOut, useSession } from '@/auth/auth-client';

export const UserMenu = ({ version }: { version?: string }) => {
	const { data } = useSession();
	const [open, setOpen] = useState(false);
	const [signingOut, setSigningOut] = useState(false);
	const menuRef = useRef<HTMLDivElement>(null);

	// Close the dropdown on outside click or Escape.
	useEffect(() => {
		if (!open) return undefined;
		const onPointerDown = (event: MouseEvent) => {
			if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
				setOpen(false);
			}
		};
		const onKeyDown = (event: KeyboardEvent) => {
			if (event.key === 'Escape') setOpen(false);
		};
		document.addEventListener('mousedown', onPointerDown);
		document.addEventListener('keydown', onKeyDown);
		return () => {
			document.removeEventListener('mousedown', onPointerDown);
			document.removeEventListener('keydown', onKeyDown);
		};
	}, [open]);

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
		<div ref={menuRef} className="relative">
			<button
				type="button"
				onClick={() => setOpen((value) => !value)}
				aria-haspopup="menu"
				aria-expanded={open}
				aria-label="User menu"
				className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-full bg-[#76b900]/15 text-xs font-semibold text-[#76b900] transition-colors hover:bg-[#76b900]/25"
			>
				{initial}
			</button>

			{open ? (
				<div
					role="menu"
					className="absolute right-0 top-full z-50 mt-2 w-56 overflow-hidden rounded-md border border-zinc-200 bg-white py-1 text-sm shadow-lg dark:border-zinc-800 dark:bg-zinc-900"
				>
					{version ? (
						<div className="px-3 py-2 text-xs text-zinc-400 dark:text-zinc-500">
							Version {version}
						</div>
					) : null}
					<div className="border-t border-zinc-100 px-3 py-2 dark:border-zinc-800">
						<div className="truncate text-xs font-medium text-zinc-400 dark:text-zinc-500">
							{fullName}
						</div>
						{roleLabel ? (
							<div className="text-[10px] text-zinc-400 dark:text-zinc-500">
								{roleLabel}
							</div>
						) : null}
					</div>
					<div className="border-t border-zinc-100 p-1 dark:border-zinc-800">
						<button
							type="button"
							onClick={handleSignOut}
							disabled={signingOut}
							className="w-full cursor-pointer rounded px-2 py-1.5 text-left text-zinc-600 transition-colors hover:bg-zinc-100 disabled:opacity-50 dark:text-zinc-300 dark:hover:bg-zinc-800"
						>
							{signingOut ? 'Signing out…' : 'Sign out'}
						</button>
					</div>
				</div>
			) : null}
		</div>
	);
};
