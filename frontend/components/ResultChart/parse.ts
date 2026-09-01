// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
	CHART_COLORS,
	CHART_TYPES,
	KPI_TONES,
	VALUE_FORMATS,
	type ChartCarouselSpec,
	type ChartCell,
	type ChartColor,
	type ChartKpi,
	type ChartSeries,
	type ChartSpec,
	type ChartType,
	type KpiOnlySpec,
	type KpiTone,
	type ValueFormat,
} from './types';

function parseJson(raw: string): unknown {
	try {
		return JSON.parse(raw);
	} catch {
		return null;
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
	return typeof value === 'string' && value.length > 0;
}

function isChartType(value: unknown): value is ChartType {
	return typeof value === 'string' && (CHART_TYPES as readonly string[]).includes(value);
}

function isChartColor(value: unknown): value is ChartColor {
	return typeof value === 'string' && (CHART_COLORS as readonly string[]).includes(value);
}

function isValueFormat(value: unknown): value is ValueFormat {
	return typeof value === 'string' && (VALUE_FORMATS as readonly string[]).includes(value);
}

function isKpiTone(value: unknown): value is KpiTone {
	return typeof value === 'string' && (KPI_TONES as readonly string[]).includes(value);
}

function isChartCell(value: unknown): value is ChartCell {
	return value === null || typeof value === 'string' || typeof value === 'number';
}

function parseSeries(value: unknown): ChartSeries | null {
	if (!isRecord(value) || !isNonEmptyString(value.key)) return null;
	const series: ChartSeries = { key: value.key };
	if (value.label !== undefined) {
		if (typeof value.label !== 'string') return null;
		series.label = value.label;
	}
	if (value.color !== undefined) {
		if (!isChartColor(value.color)) return null;
		series.color = value.color;
	}
	return series;
}

function parseKpi(value: unknown): ChartKpi | null {
	if (!isRecord(value) || !isNonEmptyString(value.label) || !isNonEmptyString(value.value)) {
		return null;
	}
	const kpi: ChartKpi = { label: value.label, value: value.value };
	if (value.sub !== undefined) {
		if (typeof value.sub !== 'string') return null;
		kpi.sub = value.sub;
	}
	if (value.tone !== undefined) {
		if (!isKpiTone(value.tone)) return null;
		kpi.tone = value.tone;
	}
	return kpi;
}

function parseDataRow(value: unknown): Record<string, ChartCell> | null {
	if (!isRecord(value)) return null;
	const row: Record<string, ChartCell> = {};
	for (const [key, cell] of Object.entries(value)) {
		if (!isChartCell(cell)) return null;
		row[key] = cell;
	}
	return row;
}

function parseChartSpecObject(value: unknown): ChartSpec | null {
	if (!isRecord(value)) return null;
	if (!isChartType(value.type) || !isNonEmptyString(value.title)) return null;
	if (!isRecord(value.x) || !isNonEmptyString(value.x.key)) return null;
	if (!Array.isArray(value.series) || value.series.length < 1 || value.series.length > 6) {
		return null;
	}
	if (!Array.isArray(value.data) || value.data.length < 1 || value.data.length > 60) {
		return null;
	}

	const series: ChartSeries[] = [];
	for (const item of value.series) {
		const parsed = parseSeries(item);
		if (!parsed) return null;
		series.push(parsed);
	}

	const data: Record<string, ChartCell>[] = [];
	for (const item of value.data) {
		const row = parseDataRow(item);
		if (!row) return null;
		data.push(row);
	}

	// A delta chart colors bars by sign with no legend, so only one series.
	if (value.type === 'delta' && series.length !== 1) return null;

	const spec: ChartSpec = {
		type: value.type,
		title: value.title,
		x: { key: value.x.key },
		series,
		data,
	};

	if (value.x.label !== undefined) {
		if (typeof value.x.label !== 'string') return null;
		spec.x.label = value.x.label;
	}

	if (value.subtitle !== undefined) {
		if (typeof value.subtitle !== 'string') return null;
		spec.subtitle = value.subtitle;
	}

	if (value.y !== undefined) {
		if (!isRecord(value.y)) return null;
		const y: NonNullable<ChartSpec['y']> = {};
		if (value.y.label !== undefined) {
			if (typeof value.y.label !== 'string') return null;
			y.label = value.y.label;
		}
		if (value.y.format !== undefined) {
			if (!isValueFormat(value.y.format)) return null;
			y.format = value.y.format;
		}
		spec.y = y;
	}

	if (value.kpis !== undefined) {
		if (!Array.isArray(value.kpis) || value.kpis.length > 4) return null;
		const kpis: ChartKpi[] = [];
		for (const item of value.kpis) {
			const kpi = parseKpi(item);
			if (!kpi) return null;
			kpis.push(kpi);
		}
		spec.kpis = kpis;
	}

	return spec;
}

/**
 * Coerce a cell value (number, or a numeric-ish string like "11,463" or "$1.2M")
 * to a number. A trailing "%" denotes a fraction, so "94%" becomes 0.94 to match
 * the `percent` format contract (data is fractions 0-1); "$", ",", and whitespace
 * are stripped without rescaling.
 */
export function toNumber(value: unknown): number | null {
	if (typeof value === 'number') return Number.isFinite(value) ? value : null;
	if (typeof value === 'string') {
		const isPercent = value.trimEnd().endsWith('%');
		const cleaned = value.replace(/[,$\s%]/g, '');
		if (cleaned === '') return null;
		const parsed = Number(cleaned);
		if (!Number.isFinite(parsed)) return null;
		return isPercent ? parsed / 100 : parsed;
	}
	return null;
}

/**
 * Parse + validate a ```chart block's JSON into a {@link ChartSpec}. Returns null
 * (so the caller can fall back to the raw block) when the JSON is malformed, fails
 * the schema, or references keys with no usable data.
 */
export function parseChartSpec(raw: string): ChartSpec | null {
	const spec = parseChartSpecObject(parseJson(raw));
	if (!spec) return null;

	// The x key must resolve on at least one row, and every series must carry at
	// least one numeric value, otherwise there is nothing meaningful to draw.
	const hasX = spec.data.some((row) => row[spec.x.key] != null && row[spec.x.key] !== '');
	if (!hasX) return null;
	const everySeriesNumeric = spec.series.every((s) =>
		spec.data.some((row) => toNumber(row[s.key]) != null),
	);
	if (!everySeriesNumeric) return null;

	return spec;
}

/** Parse + validate a ```chart-carousel block of related line charts. */
export function parseCarouselSpec(raw: string): ChartCarouselSpec | null {
	const value = parseJson(raw);
	if (!isRecord(value) || !isNonEmptyString(value.title)) return null;
	if (!Array.isArray(value.charts) || value.charts.length < 2 || value.charts.length > 12) {
		return null;
	}

	const charts: ChartCarouselSpec['charts'] = [];
	for (const item of value.charts) {
		const chart = parseChartSpec(JSON.stringify(item));
		if (!chart || chart.type !== 'line') return null;
		charts.push({ ...chart, type: 'line' });
	}

	return { title: value.title, charts };
}

/**
 * Parse a KPI-only block ({ title?, subtitle?, kpis }) with no chart axes/data.
 * Used for a single value or a one-entity result that is not worth a chart.
 */
export function parseKpiSpec(raw: string): KpiOnlySpec | null {
	const value = parseJson(raw);
	if (!isRecord(value)) return null;
	if (!Array.isArray(value.kpis) || value.kpis.length < 1 || value.kpis.length > 4) {
		return null;
	}

	const kpis: ChartKpi[] = [];
	for (const item of value.kpis) {
		const kpi = parseKpi(item);
		if (!kpi) return null;
		kpis.push(kpi);
	}

	const spec: KpiOnlySpec = { kpis };
	if (value.title !== undefined) {
		if (typeof value.title !== 'string') return null;
		spec.title = value.title;
	}
	if (value.subtitle !== undefined) {
		if (typeof value.subtitle !== 'string') return null;
		spec.subtitle = value.subtitle;
	}
	return spec;
}

/**
 * The agent sometimes emits a chart spec as a bare JSON line instead of a fenced
 * ```chart block, which would render as raw JSON. Wrap any standalone line that is
 * a valid chart (or kpi-only) spec in the matching fence so it renders regardless.
 * Lines already inside a code fence are left untouched.
 */
export function fenceBareSpecs(markdown: string): string {
	if (!markdown.includes('{')) return markdown;
	let inFence = false;
	return markdown
		.split('\n')
		.map((line) => {
			if (line.trimStart().startsWith('```')) {
				inFence = !inFence;
				return line;
			}
			if (inFence) return line;
			const trimmed = line.trim();
			if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
				if (parseCarouselSpec(trimmed)) return '```chart-carousel\n' + trimmed + '\n```';
				if (parseChartSpec(trimmed) || parseKpiSpec(trimmed)) {
					return '```chart\n' + trimmed + '\n```';
				}
			}
			return line;
		})
		.join('\n');
}
