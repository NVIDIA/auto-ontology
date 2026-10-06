// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Default and bounds of the SQL Query Timeout setting (Settings > Agent
 * Settings). They mirror `DEFAULT_`/`MIN_`/`MAX_SQL_QUERY_TIMEOUT_SECONDS` in
 * `auto_ontology/infra/feature_flags.py`, which reads the stored value before
 * every statement the text-to-SQL agent issues.
 *
 * Kept free of Prisma and Next-runtime imports: the client settings page and
 * the OpenAPI generator both import it.
 */

export const DEFAULT_SQL_QUERY_TIMEOUT_SECONDS = 30;
export const MIN_SQL_QUERY_TIMEOUT_SECONDS = 1;
export const MAX_SQL_QUERY_TIMEOUT_SECONDS = 3600;

export function isValidSqlQueryTimeout(seconds: number): boolean {
	return (
		Number.isInteger(seconds) &&
		seconds >= MIN_SQL_QUERY_TIMEOUT_SECONDS &&
		seconds <= MAX_SQL_QUERY_TIMEOUT_SECONDS
	);
}

/**
 * Read the stored timeout the way the Python reader does: a missing row, a
 * non-integer, or a value outside the bounds all mean the default, so the page
 * never shows a number the backend is not actually using.
 */
export function readSqlQueryTimeout(value: string | null | undefined): number {
	const trimmed = value?.trim() ?? '';
	if (!/^[+-]?\d+$/.test(trimmed)) return DEFAULT_SQL_QUERY_TIMEOUT_SECONDS;
	const seconds = Number(trimmed);
	return isValidSqlQueryTimeout(seconds) ? seconds : DEFAULT_SQL_QUERY_TIMEOUT_SECONDS;
}
