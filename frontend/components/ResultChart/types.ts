// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const CHART_TYPES = ['bar', 'hbar', 'line', 'area', 'grouped-bar', 'delta'] as const;
export const CHART_COLORS = ['green', 'blue', 'amber', 'red', 'neutral'] as const;
export const VALUE_FORMATS = ['number', 'compact', 'percent', 'currency'] as const;
export const KPI_TONES = ['default', 'accent', 'warn', 'alarm'] as const;

export type ChartType = (typeof CHART_TYPES)[number];
export type ChartColor = (typeof CHART_COLORS)[number];
export type ValueFormat = (typeof VALUE_FORMATS)[number];
export type KpiTone = (typeof KPI_TONES)[number];
export type ChartCell = string | number | null;

export type ChartSeries = {
	key: string;
	label?: string;
	color?: ChartColor;
};

export type ChartKpi = {
	label: string;
	value: string;
	sub?: string;
	tone?: KpiTone;
};

/**
 * The declarative chart the agent emits as a fenced ```chart JSON block. The
 * agent chooses the type, encodings, data and formatting; the frontend owns the
 * rendering style, dimensions and safe axis defaults.
 */
export type ChartSpec = {
	type: ChartType;
	title: string;
	subtitle?: string;
	x: { key: string; label?: string };
	y?: { label?: string; format?: ValueFormat };
	series: ChartSeries[];
	data: Record<string, ChartCell>[];
	kpis?: ChartKpi[];
};

/** A pageable set of related line charts emitted in a ```chart-carousel block. */
export type ChartCarouselSpec = {
	title: string;
	charts: Array<ChartSpec & { type: 'line' }>;
};

/**
 * A KPI-only block: headline tiles with no chart body. Emitted for a single
 * value or a one-entity yes/no result, where a chart would compare nothing.
 */
export type KpiOnlySpec = {
	title?: string;
	subtitle?: string;
	kpis: ChartKpi[];
};
