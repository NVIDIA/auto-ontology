// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Icon, IconName } from '@/common/icons';

export type InputsSearchProps = {
	value: string;
	onChange: (value: string) => void;
	placeholder?: string;
	disabled?: boolean;
	autoFocus?: boolean;
	className?: string;
	'aria-label'?: string;
};

/**
 * "AO-IL Inputs Search" from the design system (Filter=No): a 36px bordered
 * field with a search icon, the text in Text/Secondary at rest and Text/Primary
 * on hover or while typing, and a Close X (light-green tile with a green X on
 * hover) once there is something to clear.
 * Colours are the AO-IL Theme Light / Dark values. The design has no focus
 * state, so keyboard focus is shown with the Grey 300 - Strong token (#d3d4d6,
 * dark #3f3f46) as a border plus a 1px ring.
 */
export const InputsSearch = ({
	value,
	onChange,
	placeholder = 'Search',
	disabled = false,
	autoFocus = false,
	className = '',
	'aria-label': ariaLabel,
}: InputsSearchProps) => (
	<div
		className={`group flex h-9 items-center justify-between rounded-lg border border-[#e9eaea] bg-white px-1.5 transition-colors focus-within:border-[#d3d4d6] focus-within:ring-1 focus-within:ring-[#d3d4d6] dark:border-[#2e2e33] dark:bg-[#18181b] dark:focus-within:border-[#3f3f46] dark:focus-within:ring-[#3f3f46] ${className}`}
	>
		<div className="flex min-w-0 flex-1 items-center gap-1">
			<Icon
				name={IconName.Search}
				className="h-6 w-6 shrink-0 text-[#2b3036] dark:text-[#f0f0f3]"
				aria-hidden
			/>
			<input
				type="text"
				value={value}
				disabled={disabled}
				autoFocus={autoFocus}
				onChange={(event) => onChange(event.target.value)}
				placeholder={placeholder}
				aria-label={ariaLabel ?? placeholder}
				className={`min-w-0 flex-1 bg-transparent text-sm text-[#2b3036] outline-none placeholder:text-[#5a5e64] disabled:text-[#8d9096] disabled:placeholder:text-[#8d9096] dark:text-[#f0f0f3] dark:placeholder:text-[#b8b8c1] dark:disabled:text-[#8a8a94] dark:disabled:placeholder:text-[#8a8a94] ${
					disabled
						? ''
						: 'group-hover:placeholder:text-[#2b3036] dark:group-hover:placeholder:text-[#f0f0f3]'
				}`}
			/>
		</div>
		{value !== '' && !disabled && (
			<button
				type="button"
				onClick={() => onChange('')}
				aria-label="Clear search"
				className="h-6 w-6 shrink-0 rounded text-[#2b3036] transition-colors hover:bg-[#f1f8e5] hover:text-[#76b900] focus-visible:bg-[#f1f8e5] focus-visible:text-[#76b900] focus-visible:outline-none dark:text-[#f0f0f3] dark:hover:bg-[#8ad21a]/12 dark:hover:text-[#8ad21a] dark:focus-visible:bg-[#8ad21a]/12 dark:focus-visible:text-[#8ad21a]"
			>
				<Icon name={IconName.CloseX} className="h-6 w-6" aria-hidden />
			</button>
		)}
	</div>
);
