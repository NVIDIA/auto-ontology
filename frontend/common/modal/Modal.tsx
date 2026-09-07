// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

type ModalProps = {
	open: boolean;
	onClose: () => void;
	children: ReactNode;
	className?: string;
	overlayClassName?: string;
	align?: 'center' | 'top';
};

export const Modal = ({
	open,
	onClose,
	children,
	className = '',
	overlayClassName,
	align = 'center',
}: ModalProps) => {
	useEffect(() => {
		if (!open) return;
		const prev = document.body.style.overflow;
		document.body.style.overflow = 'hidden';
		return () => {
			document.body.style.overflow = prev;
		};
	}, [open]);

	useEffect(() => {
		if (!open) return;
		const handleKey = (e: KeyboardEvent) => {
			if (e.key === 'Escape') onClose();
		};
		document.addEventListener('keydown', handleKey);
		return () => document.removeEventListener('keydown', handleKey);
	}, [open, onClose]);

	if (!open) return null;

	return createPortal(
		<div
			className={`fixed inset-0 z-50 flex justify-center ${
				align === 'top' ? 'items-start' : 'items-center'
			} ${overlayClassName ?? (align === 'top' ? 'p-4 pt-16' : 'p-4')}`}
		>
			<div
				className="absolute inset-0 bg-black/40 transition-opacity"
				onClick={onClose}
				aria-hidden
			/>
			<div
				className={`relative rounded-xl border border-zinc-200 bg-white shadow-xl dark:border-zinc-700 dark:bg-zinc-800 ${className}`}
				role="dialog"
				aria-modal="true"
			>
				{children}
			</div>
		</div>,
		document.body,
	);
};
