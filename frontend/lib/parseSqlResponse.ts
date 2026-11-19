// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type TableRow = Record<string, string>;

export type ParsedTable = {
	columns: string[];
	rows: TableRow[];
};

/** Column names come from query results, so a `__proto__` column must stay a plain field. */
const emptyRow = (): TableRow => Object.create(null);

/**
 * Parse a JSON-stringified list of objects (the shape used when the agent
 * returns ``list[dict]`` rows) into a uniform ``ParsedTable``. Column order
 * follows the first row's key insertion order.
 */
const parseJsonTable = (text: string): ParsedTable | null => {
	let value: unknown;
	try {
		value = JSON.parse(text);
	} catch {
		return null;
	}

	// Some agent paths wrap the payload in an outer single-element list of
	// strings, e.g. `['[{"count":712}]']`. Unwrap once and retry.
	if (Array.isArray(value) && value.length === 1 && typeof value[0] === 'string') {
		return parseJsonTable(value[0]);
	}

	if (!Array.isArray(value) || value.length === 0) return null;
	if (!value.every((row) => row && typeof row === 'object' && !Array.isArray(row))) {
		return null;
	}

	const records = value as Array<Record<string, unknown>>;
	const columns = Object.keys(records[0]);
	const rows: TableRow[] = records.map((row) => {
		const out = emptyRow();
		for (const col of columns) {
			const cell = row[col];
			out[col] = cell == null ? '' : String(cell);
		}
		return out;
	});

	return { columns, rows };
};

/**
 * Parse a GitHub-flavoured markdown table (optionally wrapped in a fenced
 * code block) into a uniform ``ParsedTable``. Returns ``null`` if the text
 * does not contain a recognisable table separator.
 */
const parseMarkdownTable = (text: string): ParsedTable | null => {
	if (!text.includes('---|') && !text.includes('|---')) return null;

	const fenceMatch = text.match(/```(?:[a-zA-Z]+)?\n([\s\S]*?)\n```/);
	const body = fenceMatch ? fenceMatch[1] : text;

	const lines = body
		.split('\n')
		.map((l) => l.trim())
		.filter(Boolean);
	const separatorIdx = lines.findIndex((l) =>
		/\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?/.test(l),
	);
	if (separatorIdx <= 0) return null;

	const splitRow = (line: string) =>
		line
			.replace(/^\||\|$/g, '')
			.split('|')
			.map((c) => c.trim());

	const columns = splitRow(lines[separatorIdx - 1]).filter(Boolean);
	if (columns.length === 0) return null;

	const rows: TableRow[] = [];
	for (let i = separatorIdx + 1; i < lines.length; i++) {
		const cells = splitRow(lines[i]);
		if (cells.length !== columns.length) continue;
		const row = emptyRow();
		columns.forEach((col, idx) => {
			row[col] = cells[idx] ?? '';
		});
		rows.push(row);
	}

	if (rows.length === 0) return null;
	return { columns, rows };
};

/**
 * Parse a CSV-ish blob (the shape used when the agent returns
 * ``pd.DataFrame.to_csv(index=False)``). Naïve splitter — does not handle
 * embedded commas/quotes, but the agent rarely emits those for tabular
 * results and we'd rather fall back to the raw view than misrender.
 */
const parseCsvTable = (text: string): ParsedTable | null => {
	const lines = text
		.split('\n')
		.map((l) => l.trimEnd())
		.filter(Boolean);
	if (lines.length < 2) return null;
	if (lines.some((l) => l.includes('"'))) return null;

	const split = (line: string) => line.split(',').map((c) => c.trim());
	const columns = split(lines[0]);
	if (columns.length < 2 || columns.some((c) => !c)) return null;

	const rows: TableRow[] = [];
	for (let i = 1; i < lines.length; i++) {
		const cells = split(lines[i]);
		if (cells.length !== columns.length) return null;
		const row = emptyRow();
		columns.forEach((col, idx) => {
			row[col] = cells[idx] ?? '';
		});
		rows.push(row);
	}

	if (rows.length === 0) return null;
	return { columns, rows };
};

/**
 * Best-effort parse of the ``sql_response_from_db`` payload into a uniform
 * tabular structure. Returns ``null`` when the value cannot be recognised
 * as a table, so callers can fall back to a plain-text view.
 */
export const parseSqlResponse = (value: string | undefined | null): ParsedTable | null => {
	if (!value) return null;
	const text = value.trim();
	if (!text) return null;

	return parseJsonTable(text) ?? parseMarkdownTable(text) ?? parseCsvTable(text);
};
