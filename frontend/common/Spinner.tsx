// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

type SpinnerProps = {
	className?: string;
};

// Small inline loading indicator — a rotating ring. Shared by Toast's
// "pending" variant and any other spot that needs a lightweight, in-line
// "this is running" cue without pulling in a bigger loader.
export const Spinner = ({ className = 'h-5 w-5' }: SpinnerProps) => (
	<svg
		viewBox="0 0 24 24"
		fill="none"
		stroke="currentColor"
		strokeWidth={2}
		strokeLinecap="round"
		strokeLinejoin="round"
		aria-hidden
		className={`animate-spin ${className}`}
	>
		<circle cx="12" cy="12" r="10" opacity="0.25" />
		<path d="M22 12a10 10 0 0 0-10-10" />
	</svg>
);
