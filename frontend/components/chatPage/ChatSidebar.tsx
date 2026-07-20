// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import type { Conversation } from '@/types/chat';
import { formatDate } from '@/common/date';
import { Icon, IconName } from '@/components/icons';
import { ConfirmModal } from '@/components/modal';
import { PopoverMenu } from '@/components/PopoverMenu';

type ChatSidebarProps = {
	conversations: Conversation[];
	activeId: string | null;
	onSelect: (id: string) => void;
	onNewChat: () => void;
	onRename: (id: string, title: string) => void;
	onDelete: (id: string) => void;
	isOpen: boolean;
	onToggle: () => void;
	sidebarLoading?: boolean;
};

function ConversationItem({
	conv,
	isActive,
	onSelect,
	onRename,
	onDelete,
}: {
	conv: Conversation;
	isActive: boolean;
	onSelect: () => void;
	onRename: (title: string) => void;
	onDelete: () => void;
}) {
	const [editing, setEditing] = useState(false);
	const [confirmDelete, setConfirmDelete] = useState(false);
	const [editValue, setEditValue] = useState(conv.title);
	const inputRef = useRef<HTMLInputElement>(null);

	useEffect(() => {
		if (editing) inputRef.current?.focus();
	}, [editing]);

	const handleRenameSubmit = useCallback(() => {
		const trimmed = editValue.trim();
		if (trimmed && trimmed !== conv.title) {
			onRename(trimmed);
		}
		setEditing(false);
	}, [editValue, conv.title, onRename]);

	if (editing) {
		return (
			<li>
				<form
					onSubmit={(e) => {
						e.preventDefault();
						handleRenameSubmit();
					}}
					className="flex gap-1 px-1 py-1"
				>
					<input
						ref={inputRef}
						value={editValue}
						onChange={(e) => setEditValue(e.target.value)}
						onBlur={handleRenameSubmit}
						onKeyDown={(e) => {
							if (e.key === 'Escape') {
								setEditValue(conv.title);
								setEditing(false);
							}
						}}
						className="min-w-0 flex-1 rounded border border-zinc-300 bg-white px-2 py-1 text-sm text-black outline-none focus:border-[#76b900] dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-100 dark:focus:border-[#76b900]"
					/>
				</form>
			</li>
		);
	}

	return (
		<li className="group relative">
			<button
				type="button"
				onClick={onSelect}
				className={`w-full rounded-lg px-3 py-2 pr-8 text-left text-sm transition-colors ${
					isActive
						? 'bg-zinc-100 font-medium text-black dark:bg-zinc-800 dark:text-zinc-100'
						: 'text-zinc-700 hover:bg-zinc-100 hover:text-black dark:text-zinc-300 dark:hover:bg-zinc-800 dark:hover:text-zinc-100'
				}`}
			>
				<span className="line-clamp-1">{conv.title}</span>
				<span className="mt-0.5 block text-[10px] text-zinc-500 dark:text-zinc-400">
					{formatDate(conv.createdAt, '(DD.MM.YY)')}
				</span>
			</button>

			<PopoverMenu
				className="absolute right-1 top-1.5"
				items={[
					{
						label: 'Rename',
						icon: <Icon name={IconName.Pencil} className="h-3.5 w-3.5" />,
						onClick: () => {
							setEditValue(conv.title);
							setEditing(true);
						},
					},
					{
						label: 'Delete',
						icon: <Icon name={IconName.Trash} className="h-3.5 w-3.5" />,
						onClick: () => setConfirmDelete(true),
						danger: true,
					},
				]}
				trigger={({ toggle }) => (
					<button
						type="button"
						onClick={toggle}
						className="rounded p-1 text-zinc-400 opacity-0 transition-opacity hover:bg-zinc-200 hover:text-zinc-700 group-hover:opacity-100 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
						aria-label="Conversation options"
					>
						<Icon name={IconName.DotsVertical} className="h-4 w-4" />
					</button>
				)}
			/>

			<ConfirmModal
				open={confirmDelete}
				title="Delete conversation"
				message="Are you sure you want to delete this conversation? This action cannot be undone."
				onConfirm={() => {
					setConfirmDelete(false);
					onDelete();
				}}
				onCancel={() => setConfirmDelete(false)}
			/>
		</li>
	);
}

export const ChatSidebar = ({
	conversations,
	activeId,
	onSelect,
	onNewChat,
	onRename,
	onDelete,
	isOpen,
	onToggle,
	sidebarLoading = false,
}: ChatSidebarProps) => {
	return (
		<>
			{/* Mobile toggle */}
			<button
				type="button"
				onClick={onToggle}
				className="fixed top-3 left-3 z-30 rounded-lg bg-white p-2 shadow-md dark:bg-zinc-800 dark:shadow-zinc-900/50 lg:hidden"
				aria-label="Toggle sidebar"
			>
				<Icon name={IconName.Menu} className="h-5 w-5 text-zinc-700 dark:text-zinc-200" />
			</button>

			{/* Backdrop for mobile */}
			{isOpen && (
				<div
					className="fixed inset-0 z-20 bg-black/30 lg:hidden"
					onClick={onToggle}
					aria-hidden
				/>
			)}

			{/* Sidebar panel */}
			<aside
				className={`fixed inset-y-0 left-12 z-20 flex w-[296px] flex-col border-r border-zinc-200 bg-white transition-transform duration-200 dark:border-zinc-800 dark:bg-zinc-950 lg:static lg:translate-x-0 ${
					isOpen ? 'translate-x-0' : '-translate-x-full'
				}`}
			>
				<div className="flex h-[65px] w-full shrink-0 items-center border-b border-zinc-200 px-4 dark:border-zinc-800">
					<Link href="/" className="flex items-center gap-3">
						<Icon name={IconName.NvidiaLogo} className="h-6 w-auto text-[#76b900]" />
						<span className="text-sm font-semibold tracking-wide text-black dark:text-zinc-100">
							GSF
						</span>
					</Link>
				</div>

				<div className="px-3 py-3">
					<button
						type="button"
						onClick={onNewChat}
						className="flex w-full items-center gap-2 rounded-lg border border-zinc-300 px-3 py-2 text-sm font-medium text-black transition-colors hover:border-zinc-400 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-100 dark:hover:border-zinc-600 dark:hover:bg-zinc-800"
					>
						<span className="text-lg leading-none text-[#76b900]">+</span>
						New Chat
					</button>
				</div>

				<nav className="flex-1 overflow-y-auto px-2 pb-2">
					{sidebarLoading ? (
						<div className="flex h-full items-center justify-center py-6">
							<div
								className="h-8 w-8 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900] dark:border-zinc-700"
								role="status"
								aria-label="Loading conversations"
							/>
						</div>
					) : conversations.length === 0 ? (
						<p className="px-2 py-4 text-center text-xs text-zinc-500 dark:text-zinc-400">
							No conversations yet
						</p>
					) : (
						<ul className="space-y-0.5">
							{conversations.map((conv) => (
								<ConversationItem
									key={conv.id}
									conv={conv}
									isActive={activeId === conv.id}
									onSelect={() => onSelect(conv.id)}
									onRename={(title) => onRename(conv.id, title)}
									onDelete={() => onDelete(conv.id)}
								/>
							))}
						</ul>
					)}
				</nav>
			</aside>
		</>
	);
};
