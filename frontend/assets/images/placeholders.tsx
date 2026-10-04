// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react';

import { Icon, IconName } from '@/common/icons';

const PlaceholderFrame = ({ children }: { children: ReactNode }) => (
	<div
		className="flex h-40 w-56 flex-col items-center justify-center rounded-xl border border-dashed border-zinc-300/90 bg-zinc-50/80 dark:border-zinc-600 dark:bg-zinc-900/40"
		aria-hidden
	>
		{children}
	</div>
);

const iconClassName = 'h-12 w-12 text-disabled dark:text-zinc-600';

export const Placeholders = {
	NoConnections: function NoConnections() {
		return (
			<PlaceholderFrame>
				<svg
					className={iconClassName}
					viewBox="0 0 48 48"
					fill="none"
					xmlns="http://www.w3.org/2000/svg"
				>
					<rect
						x="8"
						y="12"
						width="32"
						height="24"
						rx="4"
						stroke="currentColor"
						strokeWidth="2"
					/>
					<path
						d="M16 20h16M16 26h10"
						stroke="currentColor"
						strokeWidth="2"
						strokeLinecap="round"
					/>
				</svg>
			</PlaceholderFrame>
		);
	},
	NoResults: function NoResults() {
		return (
			<PlaceholderFrame>
				<svg
					className={iconClassName}
					viewBox="0 0 20 20"
					fill="currentColor"
					xmlns="http://www.w3.org/2000/svg"
				>
					<path
						fillRule="evenodd"
						d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z"
						clipRule="evenodd"
					/>
				</svg>
			</PlaceholderFrame>
		);
	},
	NoRules: function NoRules() {
		return (
			<PlaceholderFrame>
				{/* The icon rules are drawn with everywhere else, taken from the icon
				    set rather than copied so the two never drift apart. */}
				<Icon name={IconName.Lightning} className={iconClassName} />
			</PlaceholderFrame>
		);
	},
	NoTerms: function NoTerms() {
		return (
			<PlaceholderFrame>
				<svg
					className={iconClassName}
					viewBox="0 0 24 24"
					fill="currentColor"
					xmlns="http://www.w3.org/2000/svg"
				>
					<path
						fillRule="evenodd"
						clipRule="evenodd"
						d="M6.084 17.777a.75.75 0 0 0 .623.178c1.438-.655 2.774-1.636 3.486-2.98.57-1.039.807-2.62.807-4.685V6.5a.5.5 0 0 0-.5-.5h-5a.5.5 0 0 0-.5.5V11.5a.5.5 0 0 0 .5.5h2.175c-.081.957-.31 1.561-.604 2.063-.334.592-1.02 1.157-1.791 1.466a.75.75 0 0 0-.308 1.174l1 1.5Zm8 0a.75.75 0 0 0 .623.178c1.438-.655 2.774-1.636 3.486-2.98.57-1.039.807-2.62.807-4.685V6.5a.5.5 0 0 0-.5-.5h-5a.5.5 0 0 0-.5.5V11.5a.5.5 0 0 0 .5.5h2.175c-.081.957-.31 1.561-.604 2.063-.334.592-1.02 1.157-1.791 1.466a.75.75 0 0 0-.308 1.174l1 1.5Z"
					/>
				</svg>
			</PlaceholderFrame>
		);
	},
};
