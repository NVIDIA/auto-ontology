// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useRef, useState } from 'react';
import { createPortal } from 'react-dom';

type TruncatedTextProps = {
	text: string;
	/** Tailwind width-constraining class applied to the truncated line. Defaults to `max-w-sm`. */
	maxWidthClass?: string;
	className?: string;
};

/**
 * Single-line truncated text that shows a popover with the full value on
 * hover — but only when the text is actually clipped (`scrollWidth` exceeds
 * `clientWidth`). Short values that already fit never trigger the popover.
 *
 * The popover is rendered via a portal into `document.body` and positioned
 * with `fixed` coordinates instead of living inside the table's scroll
 * container — an `absolute` popover nested in an `overflow-auto` ancestor
 * would otherwise expand that ancestor's scrollable area and cause the
 * whole table to jump/scroll on hover.
 */
export const TruncatedText = ({
	text,
	maxWidthClass = 'max-w-sm',
	className = '',
}: TruncatedTextProps) => {
	const spanRef = useRef<HTMLSpanElement>(null);
	const [popover, setPopover] = useState<{ top: number; left: number } | null>(null);

	const handleMouseEnter = () => {
		const el = spanRef.current;
		if (el == null || el.scrollWidth <= el.clientWidth) return;
		const rect = el.getBoundingClientRect();
		setPopover({ top: rect.bottom + 4, left: rect.left });
	};

	return (
		<>
			<span
				ref={spanRef}
				className={`block truncate ${maxWidthClass} ${className}`}
				onMouseEnter={handleMouseEnter}
				onMouseLeave={() => setPopover(null)}
			>
				{text}
			</span>
			{popover != null &&
				createPortal(
					<span
						role="tooltip"
						style={{ top: popover.top, left: popover.left }}
						className="fixed z-[1000] block w-72 max-w-[min(22rem,90vw)] whitespace-normal rounded-lg border border-zinc-200 bg-white p-2.5 text-xs leading-5 text-zinc-600 shadow-xl dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
					>
						{text}
					</span>,
					document.body,
				)}
		</>
	);
};
