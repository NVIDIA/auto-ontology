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
			<div className="flex justify-end">
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
