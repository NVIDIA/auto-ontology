// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TableType } from '@/enums/datasources';

const KNOWN_TABLE_TYPES = new Set<string>(Object.values(TableType));

/** A catalog ``table_type`` the data tree can draw. Missing and unknown values are base tables. */
export function getTableType(tableType: string | null | undefined): TableType {
	if (tableType && KNOWN_TABLE_TYPES.has(tableType)) return tableType as TableType;
	return TableType.BASE_TABLE;
}
