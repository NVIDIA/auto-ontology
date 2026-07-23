// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { IconName } from '@/common/icons';
import { DataModels, TableType } from '@/enums/datasources';

/** Shared UI metadata for catalog nodes. ``DataModels.TABLE`` and ``TableType.BASE_TABLE`` both key as ``base table``. */
export const catalogNodeInfo: Record<DataModels | TableType, { icon: IconName; title: string }> = {
	[DataModels.DB]: { icon: IconName.Database, title: 'database' },
	[DataModels.SCHEMA]: { icon: IconName.Schema, title: 'schema' },
	[DataModels.TABLE]: { icon: IconName.Table, title: 'base table' },
	[DataModels.VIEW]: { icon: IconName.View, title: 'view' },
	[DataModels.MATERIALIZED_VIEW]: { icon: IconName.MaterializedView, title: 'materialized view' },
	[DataModels.COLUMN]: { icon: IconName.Column, title: 'column' },
};
