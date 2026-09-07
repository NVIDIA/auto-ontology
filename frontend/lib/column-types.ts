// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Which columns admit a hand-edited sample value.
 *
 * Mirrors `sample_values_editable` in `gsf/utils/column_types.py`, which
 * refuses the same edits with a 422 — this only decides whether the catalog UI
 * offers the edit at all. Profiled samples arrive carrying the column's own
 * type, but an edited one is whatever text the user typed, so only a column
 * whose declared SQL type can hold that text as-is is editable.
 */

const TEXT_TYPES: ReadonlySet<string> = new Set([
	'char',
	'character',
	'varchar',
	'varchar2',
	'nchar',
	'nvarchar',
	'nvarchar2',
	'varying',
	'text',
	'ntext',
	'tinytext',
	'mediumtext',
	'longtext',
	'string',
	'clob',
	'nclob',
	'citext',
	'enum',
	// Semi-structured types: a JSONB or VARIANT value may be a bare string,
	// unlike an OBJECT or ARRAY, which is why those two are absent.
	'json',
	'jsonb',
	'variant',
	'xml',
]);

/**
 * Type names tokenize on word boundaries rather than by substring search: every
 * candidate token is matched whole, so `interval` does not read as `int` and a
 * two-word name like `character varying` still resolves.
 */
const TYPE_TOKEN = /[a-z][a-z0-9_]*/g;

/**
 * Whether a column of `dataType` accepts hand-written sample values.
 *
 * An unrecognized or missing type answers `false`, which costs that column its
 * edit affordance but never its profiled samples.
 */
export const sampleValuesEditable = (dataType: string | null | undefined): boolean => {
	if (!dataType) return false;
	const lowered = dataType.trim().toLowerCase();
	// An array or composite column holds no single string a text box could
	// stand for, whatever its element type is.
	if (lowered.endsWith('[]') || lowered.startsWith('_')) return false;
	const base = lowered.split('(')[0];
	return (base.match(TYPE_TOKEN) ?? []).some((token) => TEXT_TYPES.has(token));
};

/**
 * Note shown under a read-only sample list while the page is in edit mode, so
 * a column that stays read-only reads as deliberate rather than broken.
 */
export const sampleValuesReadOnlyHint = (dataType: string | null | undefined): string => {
	// Phrased around the type name rather than in front of it: "A integer" is
	// what an indefinite article yields for half the type names there are.
	const described = dataType?.trim() ? `typed ${dataType.trim()}` : 'with no declared type';
	return `Sample values of a column ${described} are read-only — only text-typed columns (text, varchar, json, …) can be edited.`;
};
