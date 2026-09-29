// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { PopoverAlign } from '@/enums/popover';

/** Distance between the trigger and the panel, in pixels. */
const GAP = 4;

type PanelPosition = { top: number; left?: number; right?: number; width?: number };

type PopoverProps = {
	trigger: (props: { open: boolean; toggle: (e: React.MouseEvent) => void }) => ReactNode;
	/** Panel content. `close` is for a choice that ends the interaction. */
	children: (props: { close: () => void }) => ReactNode;
	align?: PopoverAlign;
	/** The panel's own width and padding — its surface and position are set here. */
	panelClassName?: string;
	className?: string;
	/**
	 * Told whether the panel is showing, for a caller that has to react to it —
	 * dimming what the panel covers, say. Reported for every way the panel
	 * closes: the trigger, an outside click, Escape, anything that moves the
	 * trigger, and the popover itself being unmounted with the panel up.
	 */
	onOpenChange?: (open: boolean) => void;
};

/**
 * Panel anchored under its trigger, dismissed on an outside click, on Escape,
 * and on anything that moves the trigger.
 *
 * The panel is rendered via a portal into `document.body` and positioned with
 * `fixed` coordinates rather than as an `absolute` child of the trigger: an
 * absolutely positioned panel is clipped by any `overflow-hidden`/`overflow-auto`
 * ancestor, which hid it entirely when the trigger lived inside a scrolling
 * table (see the certification column in `SinglePageComposer`'s data table).
 * `Text` portals its tooltip for the same reason.
 */
export const Popover = ({
	trigger,
	children,
	align = PopoverAlign.Right,
	panelClassName = '',
	className = '',
	onOpenChange,
}: PopoverProps) => {
	// Coordinates are captured together with the open state so the panel never
	// paints a frame at a previous trigger's position.
	const [position, setPosition] = useState<PanelPosition | null>(null);
	const anchorRef = useRef<HTMLDivElement>(null);
	const panelRef = useRef<HTMLDivElement>(null);
	const open = position != null;

	// Held in a ref so an inline callback does not re-report on every render.
	const onOpenChangeRef = useRef(onOpenChange);
	useEffect(() => {
		onOpenChangeRef.current = onOpenChange;
	}, [onOpenChange]);

	// `true` on the way in and `false` from the cleanup, rather than reporting
	// `open` itself. The cleanup also runs when this component goes away, which
	// is the close a caller would otherwise never hear about: reporting `open`
	// covers every dismissal the panel performs and none of the ones performed
	// *on* it, leaving a caller that dims what the panel covers dimmed for good.
	// `RuleTagPopover` is rendered only while the search has results, so an
	// emptying result list is exactly that case.
	useEffect(() => {
		if (!open) return;

		onOpenChangeRef.current?.(true);
		return () => {
			onOpenChangeRef.current?.(false);
		};
	}, [open]);

	useEffect(() => {
		if (!open) return;

		const close = () => setPosition(null);
		const handleMouseDown = (e: MouseEvent) => {
			const target = e.target as Node;
			if (anchorRef.current?.contains(target)) return;
			if (panelRef.current?.contains(target)) return;
			close();
		};
		// Escape is taken on the way down and stopped there, so a panel opened
		// inside a modal is all it closes — the modal listens on the way back up.
		const handleKeyDown = (e: KeyboardEvent) => {
			if (e.key !== 'Escape') return;
			e.stopPropagation();
			close();
		};
		// Fixed coordinates don't follow the trigger, so dismiss instead of
		// letting the panel drift away from it. Capture phase catches scrolling
		// in nested containers rather than only the window — including the
		// panel's own scroll, which has to be left alone.
		const handleScroll = (e: Event) => {
			if (panelRef.current?.contains(e.target as Node)) return;
			close();
		};
		document.addEventListener('mousedown', handleMouseDown);
		document.addEventListener('keydown', handleKeyDown, true);
		window.addEventListener('scroll', handleScroll, true);
		window.addEventListener('resize', close);
		return () => {
			document.removeEventListener('mousedown', handleMouseDown);
			document.removeEventListener('keydown', handleKeyDown, true);
			window.removeEventListener('scroll', handleScroll, true);
			window.removeEventListener('resize', close);
		};
	}, [open]);

	const toggle = (e: React.MouseEvent) => {
		e.stopPropagation();
		const rect = e.currentTarget.getBoundingClientRect();
		setPosition((prev) => {
			if (prev != null) return null;
			const top = rect.bottom + GAP;
			return align === PopoverAlign.Stretch
				? { top, left: rect.left, width: rect.width }
				: { top, right: window.innerWidth - rect.right };
		});
	};

	return (
		<div ref={anchorRef} className={className}>
			{trigger({ open, toggle })}
			{position != null &&
				createPortal(
					<div
						ref={panelRef}
						style={position}
						className={`fixed z-[1000] rounded-lg border border-zinc-200 bg-white shadow-lg dark:border-zinc-700 dark:bg-zinc-800 ${panelClassName}`}
					>
						{children({ close: () => setPosition(null) })}
					</div>,
					document.body,
				)}
		</div>
	);
};
