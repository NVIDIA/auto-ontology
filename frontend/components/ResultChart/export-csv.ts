// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ChartCell, ChartSpec } from './types';

/** Leading characters a spreadsheet may execute as a formula. */
const FORMULA_TRIGGER = /^[=+\-@\t\r]/;

/**
 * Serialize a cell to a CSV field: neutralize spreadsheet formula injection on
 * string cells (a leading `=`/`+`/`-`/`@` text becomes an inert `'`-prefixed
 * string), then quote per RFC 4180 when the field holds a comma, quote, or
 * newline. Numeric cells pass through so real negatives are never mangled.
 */
function escapeField(value: ChartCell): string {
	if (value == null) return '';
	let text = String(value);
	if (typeof value === 'string' && FORMULA_TRIGGER.test(text)) text = `'${text}`;
	if (/[",\n\r]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
	return text;
}

/** A variation selector only modifies the character before it, which is dropped with the rest. */
const VARIATION_SELECTOR = /[\uFE0E\uFE0F]/gu;

/** Everything a filename cannot hold, minus the letters, digits, marks and joins kept below. */
const NOT_IN_FILENAME = /[^\p{L}\p{N}\p{M}\s_\-\u200C\u200D]/gu;

/** A slug made only of combining marks or join controls draws nothing in a file list. */
const VISIBLE_CHARACTER = /[\p{L}\p{N}]/u;

/**
 * Slugify a chart title into a safe `.csv` filename. Letters, digits and the
 * combining marks that belong to them are kept in every script -- `\p{L}` and
 * `\p{N}` rather than `\w`, which is ASCII-only and reduced a Korean or Japanese
 * title to `chart` -- while everything that cannot appear in a filename is
 * dropped. `\p{M}` matters as much as the letters: Devanagari vowel signs and
 * viramas are marks, so without it `हिन्दी` collapses to `हनद`. The zero-width
 * join controls matter for the same reason: Persian `می‌رود` only reads as one word
 * because of its U+200C, and dropping it changes the written form. A variation
 * selector is the opposite case -- its base character is not a letter, so it goes
 * with it -- and a title left with no letter or digit falls back to `chart`
 * rather than exporting a file whose name is invisible.
 */
function toFilename(title: string): string {
	const slug = title
		.toLowerCase()
		.trim()
		.replace(VARIATION_SELECTOR, '')
		.replace(NOT_IN_FILENAME, '')
		.replace(/[\s_]+/g, '-')
		.replace(/^[\s\u200C\u200D-]+|[\s\u200C\u200D-]+$/gu, '');
	return `${VISIBLE_CHARACTER.test(slug) ? slug : 'chart'}.csv`;
}

/**
 * Serialize a chart spec's exact data rows to an RFC 4180 CSV. Columns are the x
 * key followed by each series key; headers prefer human labels. Values are the
 * same ones the chart plots, so the export is faithful by construction.
 */
export function specToCsv(spec: ChartSpec): { filename: string; csv: string } {
	const columns = [spec.x.key, ...spec.series.map((s) => s.key)];
	const headers = [spec.x.label ?? spec.x.key, ...spec.series.map((s) => s.label ?? s.key)];

	const lines = [headers.map(escapeField).join(',')];
	for (const row of spec.data) {
		lines.push(columns.map((key) => escapeField(row[key] ?? null)).join(','));
	}

	return { filename: toFilename(spec.title), csv: lines.join('\r\n') };
}

/**
 * Trigger a browser download of the spec's CSV. Kept thin and separate from
 * {@link specToCsv} so the serialization stays pure and unit-testable.
 */
export function downloadCsv(spec: ChartSpec): void {
	const { filename, csv } = specToCsv(spec);
	// Excel opens a BOM-less UTF-8 CSV in the legacy code page, which garbles
	// every non-ASCII header and value. The byte-order mark is a property of the
	// downloaded file, not of the RFC 4180 serialization, so it belongs here and
	// not in `specToCsv`.
	const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' });
	const url = URL.createObjectURL(blob);
	const anchor = document.createElement('a');
	anchor.href = url;
	anchor.download = filename;
	anchor.click();
	URL.revokeObjectURL(url);
}
