// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const Placeholders = {
	NoConnections: function NoConnections() {
		return (
			<div
				className="flex h-40 w-56 flex-col items-center justify-center rounded-xl border border-dashed border-zinc-300/90 bg-zinc-50/80 dark:border-zinc-600 dark:bg-zinc-900/40"
				aria-hidden
			>
				<svg
					className="h-12 w-12 text-zinc-300 dark:text-zinc-600"
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
			</div>
		);
	},
	NoTerms: function NoTerms() {
		return (
			<div
				className="flex h-40 w-56 flex-col items-center justify-center rounded-xl border border-dashed border-zinc-300/90 bg-zinc-50/80 dark:border-zinc-600 dark:bg-zinc-900/40"
				aria-hidden
			>
				<svg
					className="h-12 w-12 text-zinc-300 dark:text-zinc-600"
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
			</div>
		);
	},
};
