// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TableType } from '@/enums/datasources';

export function getTableType(tableType: TableType | string | undefined): TableType | string {
	if (tableType) return tableType;
	return TableType.BASE_TABLE;
}
