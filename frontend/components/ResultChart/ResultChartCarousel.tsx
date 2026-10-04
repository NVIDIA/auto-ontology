// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { type FC, useState } from 'react';
import { ChevronLeft, ChevronRight } from '@/adapters/ui/icons';
import { ResultChart } from './ResultChart';
import type { ChartCarouselSpec } from './types';

/** A pageable set of related line charts (e.g. several peers' trends over time). */
export const ResultChartCarousel: FC<{ spec: ChartCarouselSpec }> = ({ spec }) => {
	const [index, setIndex] = useState(0);
	const count = spec.charts.length;

	// A new spec can carry fewer charts than the last, so clamp on render rather
	// than resetting the index from an effect.
	const active = Math.min(index, count - 1);

	const previous = () => setIndex((active - 1 + count) % count);
	const next = () => setIndex((active + 1) % count);

	return (
		<section className="result-chart-carousel" aria-label={spec.title}>
			<div className="result-chart-carousel-head">
				<span className="result-chart-carousel-title">{spec.title}</span>
				<div className="result-chart-carousel-controls">
					<button
						type="button"
						className="result-chart-carousel-button"
						aria-label="Previous Chart"
						title="Previous Chart"
						onClick={previous}
					>
						<ChevronLeft className="h-4 w-4" />
					</button>
					<span className="result-chart-carousel-count" aria-live="polite">
						{active + 1} / {count}
					</span>
					<button
						type="button"
						className="result-chart-carousel-button"
						aria-label="Next Chart"
						title="Next Chart"
						onClick={next}
					>
						<ChevronRight className="h-4 w-4" />
					</button>
				</div>
			</div>
			<ResultChart spec={spec.charts[active]} />
		</section>
	);
};
