// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef, useState } from 'react';
import type { User } from '@/types/auth';

const getInitials = (name: string): string => {
	const parts = name.trim().split(/\s+/);
	if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
	return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
};

export const UserAvatar = ({ user }: { user: User }) => (
	<span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-teal-600 text-[10px] font-semibold text-white">
		{getInitials(user.name)}
	</span>
);

export type UsersPickerProps = {
	allUsers: User[];
	selectedIds: Set<string>;
	onChange: (ids: Set<string>) => void;
	loading: boolean;
};

export const UsersPicker = ({ allUsers, selectedIds, onChange, loading }: UsersPickerProps) => {
	const [open, setOpen] = useState(false);
	const [search, setSearch] = useState('');
	const containerRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		const handler = (e: MouseEvent) => {
			if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
				setOpen(false);
			}
		};
		document.addEventListener('mousedown', handler);
		return () => document.removeEventListener('mousedown', handler);
	}, []);

	const filtered = allUsers.filter(
		(u) =>
			u.name.toLowerCase().includes(search.toLowerCase()) ||
			u.email.toLowerCase().includes(search.toLowerCase()),
	);

	const selectedUsers = allUsers.filter((u) => selectedIds.has(u.id));

	const toggle = (userId: string) => {
		const next = new Set(selectedIds);
		if (next.has(userId)) {
			next.delete(userId);
		} else {
			next.add(userId);
		}
		onChange(next);
	};

	return (
		<div ref={containerRef} className="relative">
			<label className="mb-1.5 block text-sm font-medium text-zinc-900 dark:text-zinc-100">
				Users
			</label>

			{/* Trigger */}
			<button
				type="button"
				onClick={() => setOpen((v) => !v)}
				className="flex min-h-[38px] w-full items-center gap-2 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-left text-sm text-zinc-700 transition-colors hover:border-zinc-400 focus:border-[#76b900] focus:outline-none focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300"
			>
				{loading ? (
					<span className="text-zinc-400">Loading users…</span>
				) : selectedUsers.length === 0 ? (
					<span className="text-zinc-400">Select users to grant access</span>
				) : (
					<span className="flex flex-wrap gap-1.5">
						{selectedUsers.map((u) => (
							<span
								key={u.id}
								className="flex items-center gap-1 rounded-full bg-zinc-100 px-2 py-0.5 text-xs font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"
							>
								<UserAvatar user={u} />
								{u.name}
							</span>
						))}
					</span>
				)}
				<svg
					className="ml-auto h-4 w-4 shrink-0 text-zinc-400"
					viewBox="0 0 20 20"
					fill="currentColor"
					aria-hidden
				>
					<path
						fillRule="evenodd"
						d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
						clipRule="evenodd"
					/>
				</svg>
			</button>

			{/* Dropdown */}
			{open && (
				<div className="absolute z-50 mt-1 w-full rounded-lg border border-zinc-200 bg-white shadow-lg dark:border-zinc-700 dark:bg-zinc-900">
					<div className="border-b border-zinc-100 p-2 dark:border-zinc-800">
						<div className="flex items-center gap-2 rounded-md border border-zinc-200 bg-zinc-50 px-2 py-1.5 dark:border-zinc-700 dark:bg-zinc-800">
							<svg
								className="h-3.5 w-3.5 text-zinc-400"
								viewBox="0 0 20 20"
								fill="currentColor"
								aria-hidden
							>
								<path
									fillRule="evenodd"
									d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z"
									clipRule="evenodd"
								/>
							</svg>
							<input
								autoFocus
								type="text"
								value={search}
								onChange={(e) => setSearch(e.target.value)}
								placeholder="Search"
								className="flex-1 bg-transparent text-sm text-zinc-700 outline-none placeholder:text-zinc-400 dark:text-zinc-300"
							/>
						</div>
					</div>
					<div className="max-h-52 overflow-y-auto p-1">
						<p className="px-2 py-1 text-xs font-medium text-zinc-400 dark:text-zinc-500">
							Users
						</p>
						{filtered.length === 0 ? (
							<p className="px-3 py-2 text-sm text-zinc-400">No users found</p>
						) : (
							filtered.map((user) => {
								const checked = selectedIds.has(user.id);
								return (
									<button
										key={user.id}
										type="button"
										onClick={() => toggle(user.id)}
										className="flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-sm hover:bg-zinc-50 dark:hover:bg-zinc-800"
									>
										<UserAvatar user={user} />
										<span className="min-w-0 flex-1">
											<span className="block truncate font-medium text-zinc-800 dark:text-zinc-200">
												{user.name}
											</span>
											<span className="block truncate text-xs text-zinc-400">
												{user.email}
											</span>
										</span>
										{checked && (
											<svg
												className="h-4 w-4 shrink-0 text-[#76b900]"
												viewBox="0 0 20 20"
												fill="currentColor"
												aria-hidden
											>
												<path
													fillRule="evenodd"
													d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z"
													clipRule="evenodd"
												/>
											</svg>
										)}
									</button>
								);
							})
						)}
					</div>
				</div>
			)}
		</div>
	);
};
