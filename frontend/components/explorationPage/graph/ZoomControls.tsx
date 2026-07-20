// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import type { Core } from 'cytoscape';

type ZoomControlsProps = {
	controller: Core | null;
};

export const ZoomControls = ({ controller }: ZoomControlsProps) => {
	const [zoom, setZoom] = useState(1);

	useEffect(() => {
		if (controller == null) return undefined;

		const updateZoom = () => setZoom(controller.zoom());
		const animationFrame = requestAnimationFrame(updateZoom);
		controller.on('zoom', updateZoom);
		return () => {
			cancelAnimationFrame(animationFrame);
			controller.off('zoom', updateZoom);
		};
	}, [controller]);

	const changeZoom = (delta: number) => {
		if (controller == null) return;
		const nextZoom = Math.min(3, Math.max(0.2, controller.zoom() + delta));
		controller.zoom({
			level: nextZoom,
			renderedPosition: {
				x: controller.width() / 2,
				y: controller.height() / 2,
			},
		});
	};

	const resetView = () => {
		if (controller == null) return;
		controller.zoom(1);
		controller.center();
	};

	return (
		<div className="flex items-center gap-2">
			<div className="flex h-10 items-center rounded-lg border border-zinc-200 bg-white shadow-md dark:border-zinc-700 dark:bg-zinc-900">
				<button
					type="button"
					onClick={() => changeZoom(-0.1)}
					className="h-full cursor-pointer px-3 text-lg text-zinc-600 hover:text-[#76b900] dark:text-zinc-300"
					aria-label="Zoom out"
				>
					−
				</button>
				<span className="w-12 text-center text-xs text-zinc-600 dark:text-zinc-300">
					{Math.round(zoom * 100)}%
				</span>
				<button
					type="button"
					onClick={() => changeZoom(0.1)}
					className="h-full cursor-pointer px-3 text-lg text-zinc-600 hover:text-[#76b900] dark:text-zinc-300"
					aria-label="Zoom in"
				>
					+
				</button>
			</div>
			<button
				type="button"
				onClick={resetView}
				className="flex h-10 w-10 cursor-pointer items-center justify-center rounded-lg border border-zinc-200 bg-white text-zinc-600 shadow-md transition-colors hover:text-[#76b900] dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300"
				aria-label="Reset view to initial position"
				title="Reset view to initial position"
			>
				<svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" stroke="currentColor">
					<path
						d="M7 3H3v4M13 3h4v4M7 17H3v-4m10 4h4v-4"
						strokeWidth="1.5"
						strokeLinecap="round"
						strokeLinejoin="round"
					/>
				</svg>
			</button>
		</div>
	);
};
