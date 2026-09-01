// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import type { ReactNode } from 'react';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import { Modal } from './Modal';

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
	className = 'min-h-[400px] w-[800px] max-w-full',
	secondaryAction,
}: ModalCreateNewItemProps) => (
	<Modal open={open} onClose={onClose} className={className}>
		<div className="flex items-center justify-between border-b border-zinc-200 px-6 py-4 dark:border-zinc-700">
			<h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">{title}</h3>
			<Button
				onClick={onClose}
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
			{children}
			<div className="flex justify-end gap-3">
				{secondaryAction && (
					<Button
						onClick={() => {
							void secondaryAction.onClick();
						}}
						disabled={secondaryAction.disabled}
						theme={ButtonTheme.Outline}
						size={Size.REGULAR}
					>
						{secondaryAction.label}
					</Button>
				)}
				<Button
					onClick={() => {
						void onSubmit();
					}}
					disabled={!canSubmit}
					theme={ButtonTheme.Primary}
					size={Size.REGULAR}
				>
					{submitLabel}
				</Button>
			</div>
		</div>
	</Modal>
);
