// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Lightweight icon shims for ResultChart (port leftover from @/adapters/ui/icons).
 */

import type { FC, SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement>;

export const ChevronDown: FC<IconProps> = (props) => (
	<svg
		viewBox="0 0 24 24"
		fill="none"
		stroke="currentColor"
		strokeWidth="2"
		strokeLinecap="round"
		strokeLinejoin="round"
		aria-hidden="true"
		{...props}
	>
		<path d="m6 9 6 6 6-6" />
	</svg>
);

export const ChevronUp: FC<IconProps> = (props) => (
	<svg
		viewBox="0 0 24 24"
		fill="none"
		stroke="currentColor"
		strokeWidth="2"
		strokeLinecap="round"
		strokeLinejoin="round"
		aria-hidden="true"
		{...props}
	>
		<path d="m18 15-6-6-6 6" />
	</svg>
);

export const ChevronLeft: FC<IconProps> = (props) => (
	<svg
		viewBox="0 0 24 24"
		fill="none"
		stroke="currentColor"
		strokeWidth="2"
		strokeLinecap="round"
		strokeLinejoin="round"
		aria-hidden="true"
		{...props}
	>
		<path d="m15 18-6-6 6-6" />
	</svg>
);

export const ChevronRight: FC<IconProps> = (props) => (
	<svg
		viewBox="0 0 24 24"
		fill="none"
		stroke="currentColor"
		strokeWidth="2"
		strokeLinecap="round"
		strokeLinejoin="round"
		aria-hidden="true"
		{...props}
	>
		<path d="m9 18 6-6-6-6" />
	</svg>
);

export const Download: FC<IconProps> = (props) => (
	<svg
		viewBox="0 0 24 24"
		fill="none"
		stroke="currentColor"
		strokeWidth="2"
		strokeLinecap="round"
		strokeLinejoin="round"
		aria-hidden="true"
		{...props}
	>
		<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
		<path d="m7 10 5 5 5-5" />
		<path d="M12 15V3" />
	</svg>
);
