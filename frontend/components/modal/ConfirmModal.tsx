// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef, type ReactNode } from 'react';
import { Modal } from './Modal';

type Tone = 'danger' | 'default';

const CONFIRM_CLASSES: Record<Tone, string> = {
	danger: 'bg-red-600 text-white hover:bg-red-700 dark:bg-red-600 dark:hover:bg-red-500',
	default: 'bg-[#76b900] text-white hover:bg-[#5e9400]',
};

export type ConfirmModalProps = {
	open: boolean;
	title: string;
	message: ReactNode;
	confirmLabel?: string;
	cancelLabel?: string;
	onConfirm: () => void | Promise<void>;
	onCancel: () => void;
	confirming?: boolean;
	error?: string | null;
	tone?: Tone;
	className?: string;
};

export const ConfirmModal = ({
	open,
	title,
	message,
	confirmLabel = 'Delete',
	cancelLabel = 'Cancel',
	onConfirm,
	onCancel,
	confirming = false,
	error = null,
	tone = 'danger',
	className = 'w-full max-w-sm',
}: ConfirmModalProps) => {
	const cancelRef = useRef<HTMLButtonElement>(null);

	useEffect(() => {
		if (open) cancelRef.current?.focus();
	}, [open]);

	return (
		<Modal open={open} onClose={onCancel} className={className}>
			<div className="flex items-center justify-between border-b border-zinc-200 px-6 py-4 dark:border-zinc-700">
				<h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
					{title}
				</h3>
				<button
					type="button"
					onClick={onCancel}
					disabled={confirming}
					className="cursor-pointer rounded-md p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-600 disabled:cursor-default disabled:opacity-50 dark:hover:bg-zinc-700 dark:hover:text-zinc-300"
					aria-label="Close"
				>
					<svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden>
						<path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
					</svg>
				</button>
			</div>
			<div className="space-y-4 p-6">
				<div className="text-sm text-zinc-700 dark:text-zinc-300">{message}</div>
				{error != null && (
					<p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
						{error}
					</p>
				)}
				<div className="flex justify-end gap-2">
					<button
						ref={cancelRef}
						type="button"
						onClick={onCancel}
						disabled={confirming}
						className="cursor-pointer rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-50 disabled:cursor-default disabled:opacity-50 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
					>
						{cancelLabel}
					</button>
					<button
						type="button"
						onClick={() => {
							void onConfirm();
						}}
						disabled={confirming}
						className={`cursor-pointer rounded-lg px-4 py-2 text-sm font-medium transition-colors ${confirming ? 'cursor-default bg-zinc-200 text-zinc-500 dark:bg-zinc-700 dark:text-zinc-400' : CONFIRM_CLASSES[tone]}`}
					>
						{confirming ? `${confirmLabel}…` : confirmLabel}
					</button>
				</div>
			</div>
		</Modal>
	);
};
