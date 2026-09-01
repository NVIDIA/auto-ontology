// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HTMLAttributes, KeyboardEvent, ReactNode } from 'react';

export type OptionCardProps = Omit<HTMLAttributes<HTMLDivElement>, 'className'> & {
	children: ReactNode;
};

/**
 * Large tile-shaped card, for grids of options such as connector types.
 * Rendered as a `div` (not a `button`) since it represents a selectable
 * option card rather than a plain action trigger.
 */
export const OptionCard = ({ children, onKeyDown, ...props }: OptionCardProps) => (
	<div
		{...props}
		role="button"
		tabIndex={0}
		onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
			if (event.key === 'Enter' || event.key === ' ') {
				event.preventDefault();
				event.currentTarget.click();
			}
			onKeyDown?.(event);
		}}
		className="flex h-[150px] cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-zinc-200 bg-white px-4 py-6 shadow-sm transition-colors hover:border-[#76b900]/50 hover:bg-[#76b900]/5 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:border-[#76b900]/40"
	>
		{children}
	</div>
);
