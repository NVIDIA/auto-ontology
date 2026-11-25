// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import {
	useCallback,
	useRef,
	useState,
	type KeyboardEvent,
	type PointerEvent,
	type ReactNode,
} from 'react';

export type SidePanelLayoutProps = {
	panel: ReactNode;
	children: ReactNode;
	panelAriaLabel?: string;
	expandAriaLabel?: string;
	collapseAriaLabel?: string;
	className?: string;
	defaultCollapsed?: boolean;
	expandedWidthClass?: string;
	collapsedWidthClass?: string;
	/** Adds the design system's drag bar on the panel's right edge to resize it. */
	resizable?: boolean;
	minWidth?: number;
	maxWidth?: number;
};

// The design system's side panel colors: page background and the divider the
// panel draws against the page and under its header.
const PANEL_BG = 'bg-white dark:bg-[#222226]';
const DIVIDER_BORDER = 'border-[#e9eaea] dark:border-[#2e2e33]';

// The drag bar's two states. Hovered — and held while dragging, so it does not
// flicker when the pointer outruns the bar — it takes the system background and
// its grips darken to the secondary text color.
const DRAG_BAR_BG =
	'bg-white hover:bg-[#f5f7f9] data-dragging:bg-[#f5f7f9] dark:bg-[#222226] dark:hover:bg-[#121214] dark:data-dragging:bg-[#121214]';
const GRIP =
	'bg-[#d3d4d6] group-hover:bg-[#5a5e64] group-data-dragging:bg-[#5a5e64] dark:bg-[#3f3f46] dark:group-hover:bg-[#b8b8c1] dark:group-data-dragging:bg-[#b8b8c1]';

const RESIZE_STEP = 16;

// Collapse Left ships with no background, so it is painted through a mask in
// the text color like the rest of the design icons.
const CollapseButton = ({ onClick, ariaLabel }: { onClick: () => void; ariaLabel: string }) => (
	<button
		type="button"
		onClick={onClick}
		aria-label={ariaLabel}
		className="flex size-6 shrink-0 items-center justify-center rounded text-[#2b3036] focus-visible:outline-2 focus-visible:outline-[#76b900] dark:text-[#f0f0f3]"
	>
		<span
			aria-hidden
			className="block size-6 bg-current [mask-image:url(/side-panel/collapse-left.svg)] [mask-position:center] [mask-repeat:no-repeat]"
		/>
	</button>
);

// Collapse Right ships with the panel background baked in, one file per color
// scheme, so both are drawn and CSS shows the one that matches.
/* eslint-disable @next/next/no-img-element -- static 24px design icons that need no optimization */
const ExpandButton = ({ onClick, ariaLabel }: { onClick: () => void; ariaLabel: string }) => (
	<button
		type="button"
		onClick={onClick}
		aria-label={ariaLabel}
		className="flex size-6 shrink-0 items-center justify-center rounded focus-visible:outline-2 focus-visible:outline-[#76b900]"
	>
		<img
			src="/side-panel/collapse-right-light.svg"
			alt=""
			width={24}
			height={24}
			className="block size-6 dark:hidden"
		/>
		<img
			src="/side-panel/collapse-right-dark.svg"
			alt=""
			width={24}
			height={24}
			className="hidden size-6 dark:block"
		/>
	</button>
);
/* eslint-enable @next/next/no-img-element */

/** The design system's drag bar: a 12px strip with two grip lines at its middle. */
const DragBar = ({
	width,
	minWidth,
	maxWidth,
	onResize,
	getStartWidth,
}: {
	width: number | null;
	minWidth: number;
	maxWidth: number;
	onResize: (width: number) => void;
	getStartWidth: () => number;
}) => {
	const drag = useRef<{ startX: number; startWidth: number } | null>(null);
	const [dragging, setDragging] = useState(false);
	const clamp = (w: number) => Math.min(maxWidth, Math.max(minWidth, w));

	const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
		e.preventDefault();
		e.currentTarget.setPointerCapture(e.pointerId);
		drag.current = { startX: e.clientX, startWidth: getStartWidth() };
		setDragging(true);
	};
	const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
		if (drag.current == null) return;
		onResize(clamp(drag.current.startWidth + e.clientX - drag.current.startX));
	};
	const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
		drag.current = null;
		setDragging(false);
		if (e.currentTarget.hasPointerCapture(e.pointerId)) {
			e.currentTarget.releasePointerCapture(e.pointerId);
		}
	};
	const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
		if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
		e.preventDefault();
		const delta = e.key === 'ArrowLeft' ? -RESIZE_STEP : RESIZE_STEP;
		onResize(clamp(getStartWidth() + delta));
	};

	return (
		<div
			role="separator"
			aria-orientation="vertical"
			aria-label="Resize panel"
			aria-valuemin={minWidth}
			aria-valuemax={maxWidth}
			aria-valuenow={width ?? undefined}
			tabIndex={0}
			data-dragging={dragging || undefined}
			onPointerDown={onPointerDown}
			onPointerMove={onPointerMove}
			onPointerUp={onPointerUp}
			onPointerCancel={onPointerUp}
			onKeyDown={onKeyDown}
			className={`group relative h-full w-3 shrink-0 cursor-col-resize touch-none border-l focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-[#76b900] ${DIVIDER_BORDER} ${DRAG_BAR_BG}`}
		>
			<span
				aria-hidden
				className={`absolute top-1/2 left-1 h-[19px] w-px -translate-y-1/2 rounded-[2px] ${GRIP}`}
			/>
			<span
				aria-hidden
				className={`absolute top-1/2 left-[7px] h-[19px] w-px -translate-y-1/2 rounded-[2px] ${GRIP}`}
			/>
		</div>
	);
};

export const SidePanelLayout = ({
	panel,
	children,
	panelAriaLabel,
	expandAriaLabel = 'Expand panel',
	collapseAriaLabel = 'Collapse panel',
	className = '',
	defaultCollapsed = false,
	expandedWidthClass = 'w-[296px]',
	collapsedWidthClass = 'w-8',
	resizable = false,
	minWidth = 200,
	maxWidth = 480,
}: SidePanelLayoutProps) => {
	const [collapsed, setCollapsed] = useState(defaultCollapsed);
	const toggleCollapsed = useCallback(() => setCollapsed((c) => !c), []);
	// Null until the user drags, so the panel keeps its class width until then.
	const [width, setWidth] = useState<number | null>(null);
	const asideRef = useRef<HTMLElement>(null);
	const getStartWidth = useCallback(() => asideRef.current?.offsetWidth ?? minWidth, [minWidth]);

	const expandedWidth = width == null ? expandedWidthClass : '';

	return (
		<div className={`flex h-full w-full bg-white dark:bg-zinc-950 ${className}`}>
			<aside
				ref={asideRef}
				// The dragged width is a runtime value, so it cannot be a class.
				style={!collapsed && width != null ? { width } : undefined}
				className={`flex h-full shrink-0 border-r font-sf ${DIVIDER_BORDER} ${PANEL_BG} ${collapsed ? collapsedWidthClass : expandedWidth}`}
			>
				{collapsed ? (
					<div className="flex h-full w-8 min-w-8 justify-center py-1">
						<ExpandButton onClick={toggleCollapsed} ariaLabel={expandAriaLabel} />
					</div>
				) : (
					<>
						<div className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
							<div
								className={`flex h-8 shrink-0 items-center justify-end border-b pr-3 pl-4 ${DIVIDER_BORDER}`}
							>
								<CollapseButton
									onClick={toggleCollapsed}
									ariaLabel={collapseAriaLabel}
								/>
							</div>
							<div
								className="flex min-h-0 flex-1 flex-col overflow-hidden"
								aria-label={panelAriaLabel}
							>
								{panel}
							</div>
						</div>
						{resizable ? (
							<DragBar
								width={width}
								minWidth={minWidth}
								maxWidth={maxWidth}
								onResize={setWidth}
								getStartWidth={getStartWidth}
							/>
						) : null}
					</>
				)}
			</aside>
			<div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">{children}</div>
		</div>
	);
};
