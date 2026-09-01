// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HTMLAttributes, KeyboardEvent } from 'react';

import { Size } from '@/enums/button';

const sizeClasses: Record<Size, string> = {
	[Size.SMALL]: 'h-7 w-7',
	[Size.REGULAR]: 'h-8 w-8',
	[Size.LARGE]: 'h-9 w-9',
};

export type ColorSwatchProps = Omit<HTMLAttributes<HTMLDivElement>, 'children' | 'className'> & {
	/** Tailwind background utility for the colour on show, e.g. `bg-sky-500`. */
	colorClass: string;
	size?: Size;
	selected?: boolean;
};

/**
 * Round, colour-filled swatch used to display and pick a single colour.
 * Rendered as a `div` (not a `button`) since it is a colour indicator rather
 * than an action trigger, mirroring how colour swatches are handled elsewhere.
 */
export const ColorSwatch = ({
	colorClass,
	size = Size.REGULAR,
	selected = false,
	onKeyDown,
	...props
}: ColorSwatchProps) => (
	<div
		{...props}
		role="button"
		tabIndex={0}
		aria-pressed={selected || undefined}
		onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
			if (event.key === 'Enter' || event.key === ' ') {
				event.preventDefault();
				event.currentTarget.click();
			}
			onKeyDown?.(event);
		}}
		className={[
			'cursor-pointer rounded-full border-2 transition hover:scale-105',
			sizeClasses[size],
			selected
				? 'border-zinc-900 ring-2 ring-[#76b900]/60 dark:border-zinc-100'
				: 'border-white/80 ring-1 ring-zinc-300/80 dark:border-zinc-900 dark:ring-zinc-700',
			colorClass,
		].join(' ')}
	/>
);
