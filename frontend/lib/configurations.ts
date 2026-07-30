// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPrisma } from '@/lib/prisma';

/** Key in the `configurations` table written by Settings > Agent Settings. */
export const VISUALIZATION_ENABLED_KEY = 'visualization_enabled';

/**
 * Global "Visualize SQL Results" flag. A missing row counts as disabled, so the
 * feature stays off until an admin opts in from Settings > Agent Settings.
 */
export async function isVisualizationEnabled(): Promise<boolean> {
	const row = await getPrisma().configuration.findUnique({
		where: { key: VISUALIZATION_ENABLED_KEY },
	});
	return row?.value === 'true';
}
