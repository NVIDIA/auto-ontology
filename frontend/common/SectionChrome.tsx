// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { createContext, useContext, useId, type ReactNode } from 'react';

/**
 * Extra controls a detail page hangs on a section's title row: an info
 * tooltip, a per-section Edit button, an action such as "+ SQL Attribute".
 * Keyed by section id. Without a provider every title renders as before.
 */
export const SectionChromeContext = createContext<((sectionId: string) => ReactNode) | null>(null);

const HEADING_CLASS = 'text-sm font-semibold text-heading dark:text-zinc-100';

export const SectionTitle = ({
	id,
	className,
	children,
}: {
	id: string;
	/** Extra classes for the heading itself, such as a bottom margin. */
	className?: string;
	children: ReactNode;
}) => {
	const chrome = useContext(SectionChromeContext);
	const extras = chrome?.(id);
	const heading = (
		<h2 className={className ? `${HEADING_CLASS} ${className}` : HEADING_CLASS}>{children}</h2>
	);
	if (extras == null) return heading;
	return (
		<div className="flex min-w-0 flex-1 items-center gap-1.5">
			{heading}
			{extras}
		</div>
	);
};

/**
 * A small "i" that explains a section on hover or keyboard focus.
 * Opens below and to the right so it stays inside a narrow side panel.
 */
export const InfoTip = ({ text, label }: { text: string; label: string }) => {
	const tipId = useId();
	return (
		<span className="group/info relative inline-flex">
			<button
				type="button"
				aria-label={`About ${label}`}
				aria-describedby={tipId}
				className="inline-flex h-4 w-4 items-center justify-center rounded-full text-secondary outline-none hover:text-heading focus-visible:ring-2 focus-visible:ring-[#76b900]/50 dark:text-zinc-400 dark:hover:text-zinc-100"
			>
				<svg viewBox="0 0 16 16" fill="none" aria-hidden className="h-3.5 w-3.5">
					<circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.3" />
					<path
						d="M8 7.2v3.8"
						stroke="currentColor"
						strokeWidth="1.4"
						strokeLinecap="round"
					/>
					<circle cx="8" cy="5" r="0.85" fill="currentColor" />
				</svg>
			</button>
			<span
				id={tipId}
				role="tooltip"
				className="pointer-events-none invisible absolute left-0 top-6 z-30 w-64 rounded-md bg-zinc-900 px-3 py-2 text-xs font-normal leading-relaxed text-white opacity-0 shadow-lg transition-opacity group-hover/info:visible group-hover/info:opacity-100 group-focus-within/info:visible group-focus-within/info:opacity-100 dark:bg-zinc-100 dark:text-zinc-900"
			>
				{text}
			</span>
		</span>
	);
};
