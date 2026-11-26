// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ButtonHTMLAttributes, ReactNode } from 'react';

import { SelectButtonTheme } from '@/enums/button';

/**
 * Unlike `Button`, a select button's appearance also depends on whether it is
 * currently the chosen option, so each theme carries its own state classes.
 */
type ThemeClasses = {
	base: string;
	selected?: string;
	unselected?: string;
	danger?: string;
};

const listItemBase = 'flex w-full cursor-pointer text-left text-sm transition-colors';
const listItemSelected = 'bg-zinc-100 font-semibold text-heading dark:bg-zinc-800';

const themeClasses: Record<SelectButtonTheme, ThemeClasses> = {
	[SelectButtonTheme.Avatar]: {
		base: 'flex h-8 w-8 items-center justify-center rounded-full bg-[#76b900] text-xs font-semibold text-light transition-colors hover:bg-[#5e9400]',
	},
	[SelectButtonTheme.ListItem]: {
		base: `${listItemBase} items-center gap-2 rounded-md px-3 py-2`,
		selected: listItemSelected,
		unselected: 'text-body hover:bg-zinc-100 dark:hover:bg-zinc-800',
		danger: 'text-error hover:bg-red-50 dark:hover:bg-error-bg',
	},
	[SelectButtonTheme.ListItemLink]: {
		base: `${listItemBase} items-center justify-between gap-2 rounded-md px-4 py-3`,
		selected: listItemSelected,
		unselected: 'text-body hover:bg-zinc-50 dark:hover:bg-zinc-800/40',
	},
	[SelectButtonTheme.ListItemTwoLine]: {
		// The extra right padding keeps the two text rows clear of the trailing
		// action button that these rows are overlaid with.
		//
		// The rows stretch to the row's width rather than shrinking to their
		// text (`items-start`), so a long value has a width to be clipped
		// against instead of running past the edge.
		base: `${listItemBase} flex-col items-stretch rounded-lg px-3 py-2 pr-8`,
		selected: listItemSelected,
		unselected: 'text-body hover:bg-zinc-100 hover:text-heading dark:hover:bg-zinc-800',
	},
	[SelectButtonTheme.SelectField]: {
		base: 'flex min-h-10 w-full cursor-pointer items-center justify-between gap-2 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-left text-sm text-body transition-colors hover:border-zinc-400 focus:border-[#76b900] focus:outline-none focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-700 dark:bg-zinc-900',
	},
	[SelectButtonTheme.Switcher]: {
		base: 'cursor-pointer rounded-md px-3 py-1.5 text-sm font-semibold transition-colors',
		selected: 'bg-white text-heading shadow-sm dark:bg-zinc-700',
		unselected: 'text-secondary hover:text-heading',
	},
};

const getStateClasses = (classes: ThemeClasses, selected: boolean, danger: boolean): string => {
	if (danger) return classes.danger ?? '';
	if (selected) return classes.selected ?? '';
	return classes.unselected ?? '';
};

export type SelectButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'className'> & {
	children: ReactNode;
	theme: SelectButtonTheme;
	selected?: boolean;
	danger?: boolean;
};

export const SelectButton = ({
	children,
	theme,
	selected = false,
	danger = false,
	type = 'button',
	...props
}: SelectButtonProps) => {
	const classes = themeClasses[theme];

	return (
		<button
			{...props}
			type={type}
			aria-pressed={selected || undefined}
			className={[classes.base, getStateClasses(classes, selected, danger)]
				.filter(Boolean)
				.join(' ')}
		>
			{children}
		</button>
	);
};
