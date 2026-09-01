// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { type ReactNode } from 'react';
import { SelectButton } from '@/common/Button';
import { Popover } from '@/common/Popover';
import { SelectButtonTheme } from '@/enums/button';

export type PopoverMenuItem = {
	label: string;
	icon?: ReactNode;
	onClick: () => void;
	danger?: boolean;
	disabled?: boolean;
};

type PopoverMenuProps = {
	items: PopoverMenuItem[];
	trigger: (props: { open: boolean; toggle: (e: React.MouseEvent) => void }) => ReactNode;
	/**
	 * Non-interactive block above the items, for a menu that has to name what
	 * its actions apply to. Its own width sets the menu's, since the panel
	 * otherwise shrinks to the widest item.
	 */
	header?: ReactNode;
	className?: string;
};

/**
 * List of actions anchored under the right edge of its trigger. See `Popover`
 * for the positioning and dismissal it is built on.
 */
export const PopoverMenu = ({ items, trigger, header, className = '' }: PopoverMenuProps) => (
	<Popover
		trigger={trigger}
		className={className}
		panelClassName="w-max min-w-32 whitespace-nowrap py-1"
	>
		{({ close }) => (
			<>
				{header != null && (
					<div className="mb-1 border-b border-zinc-100 pb-1 dark:border-zinc-800">
						{header}
					</div>
				)}
				{items.map((item) => (
					<SelectButton
						theme={SelectButtonTheme.ListItem}
						danger={item.danger}
						disabled={item.disabled}
						key={item.label}
						onClick={() => {
							close();
							item.onClick();
						}}
					>
						{item.icon}
						{item.label}
					</SelectButton>
				))}
			</>
		)}
	</Popover>
);
