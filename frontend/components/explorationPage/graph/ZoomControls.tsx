// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import type { GraphController } from '@/components/explorationPage/graph/GraphCanvas';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';

type ZoomControlsProps = {
	controller: GraphController | null;
};

export const ZoomControls = ({ controller }: ZoomControlsProps) => {
	const [zoom, setZoom] = useState(1);

	useEffect(() => {
		if (controller == null) return undefined;

		const camera = controller.getCamera();
		const updateZoom = () => setZoom(1 / camera.ratio);
		updateZoom();
		camera.on('updated', updateZoom);
		return () => {
			camera.off('updated', updateZoom);
		};
	}, [controller]);

	// Sigma's default zoom factor (1.5x per click) feels too subtle for this
	// graph — use a steeper factor so a single click noticeably changes scale.
	const ZOOM_FACTOR = 2.5;

	const zoomIn = () => {
		if (controller == null) return;
		void controller.getCamera().animatedZoom({ duration: 200, factor: ZOOM_FACTOR });
	};

	const zoomOut = () => {
		if (controller == null) return;
		void controller.getCamera().animatedUnzoom({ duration: 200, factor: ZOOM_FACTOR });
	};

	const resetView = () => {
		if (controller == null) return;
		// Re-derives the graph's frozen extent from every node's current
		// position (see `resetExtent`'s own comment in `GraphCanvas.tsx`)
		// before fitting, so a node dragged past the last-frozen box —
		// which `refresh()` alone can't recover — comes back into frame too.
		controller.resetExtent();
		void controller.getCamera().animatedReset({ duration: 200 });
	};

	return (
		<div className="flex items-center gap-2">
			<div className="flex h-10 items-center rounded-lg border border-zinc-200 bg-white shadow-md dark:border-zinc-700 dark:bg-zinc-900">
				<Button
					theme={ButtonTheme.Icon}
					size={Size.LARGE}
					iconOnly
					type="button"
					onClick={zoomOut}
					aria-label="Zoom out"
				>
					−
				</Button>
				<span className="w-12 text-center text-xs text-body dark:text-zinc-300">
					{Math.round(zoom * 100)}%
				</span>
				<Button
					theme={ButtonTheme.Icon}
					size={Size.LARGE}
					iconOnly
					type="button"
					onClick={zoomIn}
					aria-label="Zoom in"
				>
					+
				</Button>
			</div>
			<div className="overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-md dark:border-zinc-700 dark:bg-zinc-900">
				<Button
					theme={ButtonTheme.Icon}
					size={Size.LARGE}
					iconOnly
					type="button"
					onClick={resetView}
					aria-label="Reset View to Initial Position"
					title="Reset View to Initial Position"
				>
					<svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" stroke="currentColor">
						<path
							d="M7 3H3v4M13 3h4v4M7 17H3v-4m10 4h4v-4"
							strokeWidth="1.5"
							strokeLinecap="round"
							strokeLinejoin="round"
						/>
					</svg>
				</Button>
			</div>
		</div>
	);
};
