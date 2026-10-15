// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TableType } from '@/enums/datasources';

const KNOWN_TABLE_TYPES = new Set<string>(Object.values(TableType));

/**
 * A catalog icon key for a stored ``table_type``.
 *
 * Connectors leave the column null, or write the Postgres spelling in another
 * case (``BASE TABLE``, ``VIEW``). Either one is a base table or the matching
 * known type, so a tree row always has an icon. Unknown values are base tables.
 */
export function getTableType(tableType: string | null | undefined): TableType {
	const normalized = tableType?.trim().toLowerCase();
	if (normalized != null && KNOWN_TABLE_TYPES.has(normalized)) return normalized as TableType;
	return TableType.BASE_TABLE;
}
