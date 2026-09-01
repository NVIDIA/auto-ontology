// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HTMLAttributes, KeyboardEvent, MouseEvent } from 'react';

export type ToggleProps = Omit<
	HTMLAttributes<HTMLDivElement>,
	'children' | 'className' | 'onChange' | 'onClick' | 'onKeyDown'
> & {
	checked: boolean;
	disabled?: boolean;
	onChange: (checked: boolean) => void;
};

export const Toggle = ({ checked, disabled = false, onChange, ...props }: ToggleProps) => {
	// The toggle is often rendered inside clickable rows and cards, so its own
	// events must never reach the surrounding element.
	const toggle = (event: MouseEvent<HTMLDivElement> | KeyboardEvent<HTMLDivElement>) => {
		event.stopPropagation();
		onChange(!checked);
	};

	const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
		if (event.key !== ' ' && event.key !== 'Enter') return;
		event.preventDefault();
		toggle(event);
	};

	return (
		<div
			{...props}
			role="switch"
			aria-checked={checked}
			aria-disabled={disabled || undefined}
			tabIndex={disabled ? -1 : 0}
			onClick={disabled ? undefined : toggle}
			onKeyDown={disabled ? undefined : handleKeyDown}
			className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors focus-visible:ring-2 focus-visible:ring-[#76b900]/40 focus-visible:outline-none ${
				disabled ? 'pointer-events-none cursor-default opacity-50' : 'cursor-pointer'
			} ${checked ? 'bg-[#76b900]' : 'bg-zinc-300 dark:bg-zinc-600'}`}
		>
			<span
				className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
					checked ? 'translate-x-5' : 'translate-x-0.5'
				}`}
			/>
		</div>
	);
};
