// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { IconName } from '@/components/icons';
import { DataModels, TableType } from '@/enums/datasources';

const tableTypeToCatalogKind: Record<TableType, DataModels> = {
	[TableType.BASE_TABLE]: DataModels.TABLE,
	[TableType.VIEW]: DataModels.VIEW,
	[TableType.MATERIALIZED_VIEW]: DataModels.MATERIALIZED_VIEW,
};

export const catalogNodeInfo: Record<DataModels, { icon: IconName; title: string }> = {
	[DataModels.DB]: { icon: IconName.Database, title: 'database' },
	[DataModels.SCHEMA]: { icon: IconName.Schema, title: 'schema' },
	[DataModels.TABLE]: { icon: IconName.Table, title: 'table' },
	[DataModels.VIEW]: { icon: IconName.View, title: 'view' },
	[DataModels.MATERIALIZED_VIEW]: { icon: IconName.MaterializedView, title: 'materialized view' },
	[DataModels.COLUMN]: { icon: IconName.Column, title: 'column' },
};

export function catalogKindForTableType(tableType: TableType): DataModels {
	return tableTypeToCatalogKind[tableType] ?? DataModels.TABLE;
}
