// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef, type ReactNode } from 'react';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import { Modal } from './Modal';

type Tone = 'danger' | 'default';

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
				<Button
					onClick={onCancel}
					disabled={confirming}
					theme={ButtonTheme.Icon}
					size={Size.SMALL}
					iconOnly
					aria-label="Close"
				>
					<svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden>
						<path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
					</svg>
				</Button>
			</div>
			<div className="space-y-4 p-6">
				<div className="wrap-anywhere text-sm text-zinc-700 dark:text-zinc-300">
					{message}
				</div>
				{error != null && (
					<p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
						{error}
					</p>
				)}
				<div className="flex justify-end gap-2">
					<Button
						ref={cancelRef}
						onClick={onCancel}
						disabled={confirming}
						theme={ButtonTheme.Secondary}
						size={Size.REGULAR}
					>
						{cancelLabel}
					</Button>
					<Button
						onClick={() => {
							void onConfirm();
						}}
						loading={confirming}
						theme={tone === 'danger' ? ButtonTheme.Danger : ButtonTheme.Primary}
						size={Size.REGULAR}
					>
						{confirming ? `${confirmLabel}…` : confirmLabel}
					</Button>
				</div>
			</div>
		</Modal>
	);
};
