// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import type { ReactNode } from 'react';
import { Modal } from './Modal';

type Accent = 'emerald' | 'teal';

const ACCENT_CLASSES: Record<Accent, string> = {
	emerald: 'bg-[#76b900] text-white hover:bg-[#5e9400]',
	teal: 'bg-teal-600 text-white hover:bg-teal-700 dark:bg-teal-600 dark:hover:bg-teal-500',
};

export type ModalSecondaryAction = {
	label: string;
	onClick: () => void | Promise<void>;
	disabled?: boolean;
};

export type ModalCreateNewItemProps = {
	open: boolean;
	onClose: () => void;
	title: string;
	submitLabel: string;
	onSubmit: () => void | Promise<void>;
	canSubmit: boolean;
	children: ReactNode;
	accent?: Accent;
	className?: string;
	/** Optional secondary button rendered left of the submit button, e.g. "Validate SQL". */
	secondaryAction?: ModalSecondaryAction;
};

export const ModalCreateNewItem = ({
	open,
	onClose,
	title,
	submitLabel,
	onSubmit,
	canSubmit,
	children,
	accent = 'emerald',
	className = 'min-h-[400px] w-[800px] max-w-full',
	secondaryAction,
}: ModalCreateNewItemProps) => (
	<Modal open={open} onClose={onClose} className={className}>
		<div className="flex items-center justify-between border-b border-zinc-200 px-6 py-4 dark:border-zinc-700">
			<h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">{title}</h3>
			<button
				type="button"
				onClick={onClose}
				className="cursor-pointer rounded-md p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-600 dark:hover:bg-zinc-700 dark:hover:text-zinc-300"
				aria-label="Close"
			>
				<svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden>
					<path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
				</svg>
			</button>
		</div>
		<div className="space-y-4 p-6">
			{children}
			<div className="flex justify-end gap-3">
				{secondaryAction && (
					<button
						type="button"
						onClick={() => {
							void secondaryAction.onClick();
						}}
						disabled={secondaryAction.disabled}
						className={`cursor-pointer rounded-lg border bg-white px-4 py-2 text-sm font-medium transition-colors hover:bg-zinc-50 disabled:cursor-default disabled:opacity-50 dark:bg-zinc-900 dark:hover:bg-zinc-800 ${secondaryAction.disabled ? 'border-zinc-300 text-zinc-700 dark:border-zinc-600 dark:text-zinc-300' : 'border-[#76b900] text-[#76b900] dark:border-[#76b900] dark:text-[#76b900]'}`}
					>
						{secondaryAction.label}
					</button>
				)}
				<button
					type="button"
					onClick={() => {
						void onSubmit();
					}}
					disabled={!canSubmit}
					className={`cursor-pointer rounded-lg px-4 py-2 text-sm font-medium transition-colors ${canSubmit ? ACCENT_CLASSES[accent] : 'cursor-default bg-zinc-200 text-zinc-500 dark:bg-zinc-700 dark:text-zinc-400'}`}
				>
					{submitLabel}
				</button>
			</div>
		</div>
	</Modal>
);
