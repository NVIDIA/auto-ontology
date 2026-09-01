// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useRef, useState, type ReactNode } from 'react';
import { Tooltip } from '@nvidia/foundations-react-core';
import { TextVariant } from '@/enums/text';

type TextElement = 'span' | 'p' | 'div' | 'h1' | 'h2' | 'h3' | 'h4';

type TextProps = {
	/**
	 * Value revealed in the tooltip. Defaults to the rendered text content, so
	 * it only has to be passed when the two differ — a cell that renders a badge
	 * but should reveal the raw value, for instance. An empty string counts as
	 * not passed, since callers routinely default it to one.
	 */
	text?: string;
	/** Rendered on the truncated line. Defaults to `text`. */
	children?: ReactNode;
	/**
	 * Visible lines before the value is clipped. `1` (the default) keeps it on a
	 * single line; higher values let it wrap and clamp at the given line count.
	 */
	lines?: 1 | 2 | 3;
	/** Element to render, for headings and paragraphs that need their own tag. */
	as?: TextElement;
	/** Role the value plays on the page. Defaults to the surrounding typography. */
	variant?: TextVariant;
	/** Claims the free space of a flex row, for a value followed by trailing controls. */
	fill?: boolean;
	/**
	 * Puts a clipped line in the tab order so its tooltip can be opened without
	 * a pointer. Off by default because the line is usually already inside a
	 * button or a link, where a focusable child is both a redundant tab stop
	 * and invalid content.
	 */
	focusable?: boolean;
};

// Tailwind only emits utilities it can see spelled out in the source, so these
// cannot be assembled from `lines` or `variant` at runtime.
//
// Wrapped values keep their own line breaks and break mid-word, because a
// clamped paragraph is where pasted URLs and identifiers end up.
const CLAMP_CLASS = {
	1: 'block truncate',
	2: 'line-clamp-2 whitespace-pre-wrap wrap-anywhere',
	3: 'line-clamp-3 whitespace-pre-wrap wrap-anywhere',
} as const;

const VARIANT_CLASS: Record<TextVariant, string> = {
	[TextVariant.Inherit]: '',
	[TextVariant.PageTitle]:
		'text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100',
	[TextVariant.CardTitle]:
		'text-base font-semibold tracking-tight text-zinc-900 dark:text-zinc-100',
	[TextVariant.Heading]: 'text-sm font-semibold text-zinc-900 dark:text-zinc-100',
	[TextVariant.Subheading]: 'text-sm font-medium text-zinc-900 dark:text-zinc-100',
	[TextVariant.Body]: 'text-sm text-zinc-600 dark:text-zinc-300',
	[TextVariant.Strong]: 'font-medium',
	[TextVariant.Caption]: 'text-xs text-zinc-400 dark:text-zinc-500',
	[TextVariant.Detail]: 'text-xs leading-5 text-zinc-500 dark:text-zinc-400',
	[TextVariant.Label]: 'text-xs font-semibold text-zinc-500 dark:text-zinc-400',
	[TextVariant.Overline]:
		'text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400',
};
const TOOLTIP_CLASS =
	'z-[1000] block max-h-[60vh] max-w-[min(500px,90vw)] overflow-y-auto whitespace-pre-wrap wrap-anywhere rounded-lg border border-zinc-200 bg-white p-2 text-xs leading-5 text-zinc-600 shadow-xl dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300';

const cx = (...classes: (string | false | undefined)[]) => classes.filter(Boolean).join(' ');
export const Text = ({
	text,
	children,
	lines = 1,
	as = 'span',
	variant = TextVariant.Inherit,
	fill = false,
	focusable = false,
}: TextProps) => {
	const elementRef = useRef<HTMLSpanElement | null>(null);
	const [tooltip, setTooltip] = useState('');
	// Every supported tag renders a plain `HTMLElement`, so narrowing the union
	// to one member spares JSX from reconciling their prop types.
	const Element = as as 'span';

	const measure = useCallback(() => {
		const el = elementRef.current;
		if (el == null) return;
		// A clamped value overflows downwards rather than sideways. The
		// one-pixel slack absorbs sub-pixel line heights, which otherwise
		// report every clamped element as overflowing.
		const clipped =
			lines > 1 ? el.scrollHeight > el.clientHeight + 1 : el.scrollWidth > el.clientWidth;
		setTooltip(clipped ? text?.trim() || el.textContent || '' : '');
	}, [text, lines]);

	// Measured as the node attaches, so the tooltip parts are only mounted
	// around a value that has something to reveal, and again on the way in,
	// because the column can have been resized — or a webfont swapped in —
	// since that first reading. Measuring on entry is what makes an observer
	// unnecessary: the answer is only ever needed under the pointer, and there
	// it is always taken fresh.
	const attach = useCallback(
		(el: HTMLSpanElement | null) => {
			elementRef.current = el;
			if (el != null) measure();
		},
		[measure],
	);

	const line = (
		<Element
			ref={attach}
			onPointerEnter={measure}
			onFocus={measure}
			// An intact line has nothing to reveal, so it stays out of the tab
			// order even when the caller asks for focus.
			tabIndex={focusable && tooltip !== '' ? 0 : undefined}
			className={cx('min-w-0', fill && 'flex-1', CLAMP_CLASS[lines], VARIANT_CLASS[variant])}
		>
			{children ?? text}
		</Element>
	);

	// An intact value renders as a bare element. Wrapping it anyway would cost
	// a tooltip root, a popper and a portal per line, and most of the lines on
	// a page — tree rows, tags, headings, table cells that comfortably fit —
	// are never clipped.
	if (tooltip === '') return line;

	// Open delay and skip window come from the `TooltipProvider` the root layout
	// puts around the app, so they stay consistent with every other tooltip.
	return (
		<Tooltip slotContent={tooltip} side="bottom" align="start" className={TOOLTIP_CLASS}>
			{line}
		</Tooltip>
	);
};
