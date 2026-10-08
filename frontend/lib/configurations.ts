// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPrisma } from '@/lib/prisma';

/** Key in the `configurations` table written by Settings > Agent Settings. */
export const VISUALIZATION_ENABLED_KEY = 'visualization_enabled';

/**
 * Key written by the SQL Query Timeout field on Settings > Agent Settings. Its
 * default, bounds and parser live in `@/lib/sqlQueryTimeout`, which is free of
 * Prisma so the settings page and the OpenAPI generator can import them.
 */
export const SQL_QUERY_TIMEOUT_SECONDS_KEY = 'sql_query_timeout_seconds';

/** Key written by the Distinct Value Scanning toggle on Settings > Semantic Compilation. */
export const DISTINCT_VALUE_PROBING_ENABLED_KEY = 'distinct_value_probing_enabled';

/** Key written by the PII Detection toggle on Settings > PII Settings. */
export const PII_DETECTION_ENABLED_KEY = 'pii_detection_enabled';

/**
 * Read an opt-out flag the way `auto_ontology/infra/feature_flags.py` does.
 *
 * Both layers read the same row, so they have to canonicalize it the same way
 * or they disagree: the Python reader strips and lowercases, so a stored
 * `"FALSE"` or `" false "` turns the feature off there while a bare
 * `!== 'false'` here would render the toggle as on. Only `"false"` disables;
 * an absent row or an unrecognized value counts as enabled, matching that
 * reader's `default=True`.
 */
export function readOptOutFlag(value: string | null | undefined): boolean {
	return value?.trim().toLowerCase() !== 'false';
}

/**
 * Read an opt-in flag the way `auto_ontology/infra/feature_flags.py` does with
 * `default=False`. Only `"true"` enables; an absent row or an unrecognized
 * value counts as disabled, matching that reader's fallback.
 */
export function readOptInFlag(value: string | null | undefined): boolean {
	return value?.trim().toLowerCase() === 'true';
}

/**
 * Global "Visualize SQL Results" flag. A missing row counts as enabled, so the
 * feature is on by default until an admin turns it off from
 * Settings > Agent Settings.
 */
export async function isVisualizationEnabled(): Promise<boolean> {
	const row = await getPrisma().configuration.findUnique({
		where: { key: VISUALIZATION_ENABLED_KEY },
	});
	return row?.value !== 'false';
}

/**
 * Global "PII Detection" flag. A missing row counts as disabled, so ingest
 * skips classification until an admin turns it on from Settings > PII
 * Settings. Canonicalized through `readOptInFlag` so this cannot disagree
 * with the ingestion service about what the stored value means.
 */
export async function isPiiDetectionEnabled(): Promise<boolean> {
	const row = await getPrisma().configuration.findUnique({
		where: { key: PII_DETECTION_ENABLED_KEY },
	});
	return readOptInFlag(row?.value);
}
