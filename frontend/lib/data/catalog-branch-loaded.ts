// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Database } from '@/types/datasources';

function splitFocusSegments(focusId: string): string[] {
	return focusId.split('|').filter((segment) => segment.length > 0);
}

/** True when `databases` already contains enough tree data to render `focusId` without fetching catalog APIs. */
export function isCatalogBranchLoadedForFocus(databases: Database[], focusId: string): boolean {
	const segments = splitFocusSegments(focusId);
	const databaseId = segments[0];
	if (!databaseId) return false;

	const database = databases.find((entry) => entry.id === databaseId);
	if (!database) return false;

	if (segments.length === 1) {
		return database.schemas.length > 0 || database.num_of_schemas === 0;
	}

	const schemaId = segments[1];
	const schema = database.schemas.find((entry) => entry.id === schemaId);
	if (!schema) return false;

	if (segments.length === 2) {
		return schema.tables.length > 0 || schema.tables_count === 0;
	}

	const tableId = segments[2];
	const table = schema.tables.find((entry) => entry.id === tableId);
	if (!table) return false;

	const needsColumns = table.columns_count > 0;
	return !needsColumns || table.columns.length > 0;
}
