// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPrisma } from '@/lib/prisma';

/** Key in the `configurations` table written by Settings > Agent Settings. */
export const VISUALIZATION_ENABLED_KEY = 'visualization_enabled';

/** Key written by the Distinct Value Scanning toggle on Settings > Semantic Compilation. */
export const DISTINCT_VALUE_PROBING_ENABLED_KEY = 'distinct_value_probing_enabled';

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
