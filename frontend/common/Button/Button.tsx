// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { forwardRef, type ButtonHTMLAttributes } from 'react';

import { ButtonTheme, Size } from '@/enums/button';

const themeClasses: Record<ButtonTheme, string> = {
	[ButtonTheme.Primary]:
		'border border-transparent bg-[#76b900] text-white hover:bg-[#5e9400] disabled:bg-zinc-100 disabled:text-zinc-400 dark:disabled:bg-zinc-800 dark:disabled:text-zinc-500',
	[ButtonTheme.Secondary]:
		'border border-zinc-300 bg-white text-zinc-700 hover:border-[#76b900] hover:bg-[#76b900]/10 hover:text-[#5e9400] dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-[#76b900]/20 dark:hover:text-[#a3d63a] disabled:border-transparent disabled:bg-zinc-100 disabled:text-zinc-400 dark:disabled:bg-zinc-800 dark:disabled:text-zinc-500',
	[ButtonTheme.Outline]:
		'border border-[#76b900] bg-white text-[#5e9400] hover:bg-[#5e9400] hover:text-white dark:bg-zinc-900 dark:hover:bg-[#5e9400] disabled:border-zinc-300 disabled:bg-transparent disabled:text-zinc-400 dark:disabled:border-zinc-600 dark:disabled:text-zinc-500',
	[ButtonTheme.Danger]:
		'border border-transparent bg-red-600 text-white hover:bg-red-700 disabled:bg-zinc-100 disabled:text-zinc-400 dark:disabled:bg-zinc-800 dark:disabled:text-zinc-500',
	[ButtonTheme.DangerOutline]:
		'border border-red-500 bg-white text-red-600 hover:bg-red-600 hover:text-white dark:border-red-500 dark:bg-zinc-900 dark:text-red-400 dark:hover:bg-red-600 disabled:border-zinc-300 disabled:bg-white disabled:text-zinc-400 dark:disabled:border-zinc-600 dark:disabled:bg-zinc-900 dark:disabled:text-zinc-500',
	[ButtonTheme.DangerSubtle]:
		'border border-red-300 bg-white text-red-600 hover:bg-red-50 dark:border-red-800 dark:bg-zinc-900 dark:text-red-400 dark:hover:bg-red-950/30 disabled:border-zinc-300 disabled:bg-white disabled:text-zinc-400 dark:disabled:border-zinc-600 dark:disabled:bg-zinc-900 dark:disabled:text-zinc-500',
	[ButtonTheme.Minimal]:
		'border border-transparent bg-transparent text-zinc-600 hover:bg-[#76b900]/10 hover:text-[#5e9400] dark:text-zinc-300 dark:hover:bg-[#76b900]/20 dark:hover:text-[#a3d63a] disabled:text-zinc-400 disabled:hover:bg-transparent dark:disabled:text-zinc-500',
	[ButtonTheme.Icon]:
		'border border-transparent bg-transparent text-zinc-400 hover:bg-[#76b900]/10 hover:text-[#5e9400] dark:hover:bg-[#76b900]/20 dark:hover:text-[#a3d63a] disabled:text-zinc-300 disabled:hover:bg-transparent dark:disabled:text-zinc-600',
	[ButtonTheme.IconNeutral]:
		'border border-transparent bg-transparent text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-700 dark:hover:text-zinc-200 disabled:text-zinc-300 disabled:hover:bg-transparent dark:disabled:text-zinc-600',
	[ButtonTheme.IconDanger]:
		'border border-transparent bg-transparent text-zinc-500 hover:text-red-600 dark:text-zinc-400 dark:hover:text-red-400 disabled:text-zinc-300 disabled:hover:bg-transparent dark:disabled:text-zinc-600',
	[ButtonTheme.Soft]:
		'border border-[#76b900]/40 bg-white text-[#4d7a00] hover:bg-[#76b900]/10 dark:border-[#76b900]/40 dark:bg-zinc-900 dark:text-[#a3d63a] dark:hover:bg-[#76b900]/15 disabled:border-zinc-300 disabled:bg-white disabled:text-zinc-400 dark:disabled:border-zinc-600 dark:disabled:bg-zinc-900 dark:disabled:text-zinc-500',
};

const sizeClasses: Record<Size, string> = {
	[Size.SMALL]: 'h-[26px] text-xs',
	[Size.REGULAR]: 'h-9 text-sm',
	[Size.LARGE]: 'h-10 text-sm',
};

const radiusClasses: Record<Size, string> = {
	[Size.SMALL]: 'rounded-md',
	[Size.REGULAR]: 'rounded-lg',
	[Size.LARGE]: 'rounded-xl',
};

const iconOnlySizeClasses: Record<Size, string> = {
	[Size.SMALL]: 'w-[26px]',
	[Size.REGULAR]: 'w-9',
	[Size.LARGE]: 'w-10',
};

export type IconPosition = 'left' | 'right';

const setPaddingClasses = (size: Size, noPadding: boolean): string => {
	if (noPadding) return 'p-0';
	return size === Size.SMALL ? 'px-2' : 'px-4';
};

export type ButtonProps = {
	theme?: ButtonTheme;
	size?: Size;
	loading?: boolean;
	iconPosition?: IconPosition;
	iconOnly?: boolean;
	noPadding?: boolean;
	rounded?: boolean;
	full?: boolean;
	shadow?: boolean;
};

export const Button = forwardRef<
	HTMLButtonElement,
	ButtonProps & Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'className'>
>(
	(
		{
			children,
			theme = ButtonTheme.Primary,
			size = Size.REGULAR,
			loading = false,
			iconPosition,
			iconOnly = false,
			noPadding = false,
			rounded = false,
			full = false,
			shadow = false,
			disabled = false,
			type = 'button',
			...props
		},
		ref,
	) => (
		<button
			{...props}
			ref={ref}
			type={type}
			disabled={disabled || loading}
			aria-busy={loading || undefined}
			className={[
				'inline-flex cursor-pointer items-center justify-center whitespace-nowrap font-medium transition-colors disabled:cursor-default disabled:opacity-100',
				themeClasses[theme],
				sizeClasses[size],
				rounded ? 'rounded-full' : radiusClasses[size],
				iconOnly ? iconOnlySizeClasses[size] : setPaddingClasses(size, noPadding),
				iconOnly && 'p-0',
				iconPosition && 'gap-1.5',
				full && 'w-full',
				shadow && 'shadow-sm',
			]
				.filter(Boolean)
				.join(' ')}
		>
			{children}
		</button>
	),
);

Button.displayName = 'Button';
