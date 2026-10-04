// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';

export type SearchInputProps = {
	value: string;
	onChange: (value: string) => void;
	placeholder?: string;
	className?: string;
	autoFocus?: boolean;
	'aria-label'?: string;
};

export const SearchInput = ({
	value,
	onChange,
	placeholder = 'Search',
	className = '',
	autoFocus = false,
	'aria-label': ariaLabel,
}: SearchInputProps) => (
	<div
		className={`flex items-center gap-2 rounded-lg border border-zinc-300 bg-white px-3 py-2 transition-colors focus-within:border-[#76b900] focus-within:ring-2 focus-within:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 ${className}`}
	>
		<svg
			className="h-4 w-4 shrink-0 text-secondary dark:text-zinc-400"
			viewBox="0 0 20 20"
			fill="currentColor"
			aria-hidden
		>
			<path
				fillRule="evenodd"
				d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z"
				clipRule="evenodd"
			/>
		</svg>
		<input
			type="text"
			value={value}
			onChange={(e) => onChange(e.target.value)}
			placeholder={placeholder}
			autoFocus={autoFocus}
			aria-label={ariaLabel ?? placeholder}
			className="flex-1 bg-transparent text-sm text-body outline-none placeholder:text-secondary dark:text-zinc-300 dark:placeholder:text-zinc-500"
		/>
		{value !== '' && (
			<Button
				theme={ButtonTheme.IconNeutral}
				size={Size.SMALL}
				iconOnly
				rounded
				type="button"
				onClick={() => onChange('')}
				aria-label="Clear search"
			>
				<svg className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden>
					<path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
				</svg>
			</Button>
		)}
	</div>
);
