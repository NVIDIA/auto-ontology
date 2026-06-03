// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useSyncExternalStore } from 'react';
import { createPortal } from 'react-dom';

type ToastVariant = 'error' | 'success' | 'info' | 'pending';

type ToastProps = {
	open: boolean;
	message: string;
	title?: string;
	variant?: ToastVariant;
	onClose: () => void;
};

// Mirror the Omnichat AlertInline visual language: round coloured icon,
// optional bold title, message with capitalised first letter, and a close
// button on the right — all inside a white card pinned to the bottom of
// the viewport (matches `Snackbar` from omnichat's Alert/styled.tsx).

const iconColorClass: Record<ToastVariant, string> = {
	error: 'text-red-600 dark:text-red-400',
	success: 'text-emerald-600 dark:text-emerald-400',
	info: 'text-sky-600 dark:text-sky-400',
	pending: 'text-[#76b900]',
};

const VariantIcon = ({ variant }: { variant: ToastVariant }) => {
	const className = `h-5 w-5 shrink-0 ${iconColorClass[variant]}`;
	const stroke = {
		viewBox: '0 0 24 24',
		fill: 'none',
		stroke: 'currentColor',
		strokeWidth: 2,
		strokeLinecap: 'round' as const,
		strokeLinejoin: 'round' as const,
		'aria-hidden': true,
		className,
	};

	if (variant === 'success') {
		return (
			<svg {...stroke}>
				<circle cx="12" cy="12" r="10" />
				<path d="M8 12l2.5 2.5L16 9" />
			</svg>
		);
	}
	if (variant === 'info') {
		return (
			<svg {...stroke}>
				<circle cx="12" cy="12" r="10" />
				<line x1="12" y1="16" x2="12" y2="12" />
				<line x1="12" y1="8" x2="12.01" y2="8" />
			</svg>
		);
	}
	if (variant === 'pending') {
		return (
			<svg {...stroke} className={`${className} animate-spin`}>
				<circle cx="12" cy="12" r="10" opacity="0.25" />
				<path d="M22 12a10 10 0 0 0-10-10" />
			</svg>
		);
	}
	return (
		<svg {...stroke}>
			<circle cx="12" cy="12" r="10" />
			<line x1="12" y1="8" x2="12" y2="12" />
			<line x1="12" y1="16" x2="12.01" y2="16" />
		</svg>
	);
};

const defaultTitle: Record<ToastVariant, string | undefined> = {
	error: 'Error',
	success: undefined,
	info: undefined,
	pending: undefined,
};

const capitalizeFirst = (value: string): string =>
	value.length === 0 ? value : value.charAt(0).toUpperCase() + value.slice(1);

// Track hydration via useSyncExternalStore so the snapshot differs between
// server (false) and client (true) without a setState-in-effect.
const subscribeHydration = () => () => {};
const getHydratedSnapshot = () => true;
const getHydratedServerSnapshot = () => false;

export const Toast = ({ open, message, title, variant = 'error', onClose }: ToastProps) => {
	const mounted = useSyncExternalStore(
		subscribeHydration,
		getHydratedSnapshot,
		getHydratedServerSnapshot,
	);

	if (!mounted || !open) return null;

	const resolvedTitle = title ?? defaultTitle[variant];

	return createPortal(
		<div
			role="status"
			aria-live="polite"
			className="pointer-events-none fixed inset-x-0 bottom-[3%] z-[100] flex justify-center px-4"
		>
			<div className="pointer-events-auto flex max-w-md items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2 shadow-lg dark:border-zinc-700 dark:bg-zinc-800">
				<VariantIcon variant={variant} />
				{resolvedTitle ? (
					<span className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{resolvedTitle}
					</span>
				) : null}
				<span className="text-sm text-zinc-800 dark:text-zinc-200">
					{capitalizeFirst(message)}
				</span>
				<button
					type="button"
					onClick={onClose}
					aria-label="Dismiss notification"
					className="ml-2 rounded p-1 text-zinc-500 opacity-80 transition-opacity hover:opacity-100 dark:text-zinc-400"
				>
					<svg
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						strokeWidth={2}
						strokeLinecap="round"
						strokeLinejoin="round"
						className="h-4 w-4"
					>
						<line x1="18" y1="6" x2="6" y2="18" />
						<line x1="6" y1="6" x2="18" y2="18" />
					</svg>
				</button>
			</div>
		</div>,
		document.body,
	);
};
