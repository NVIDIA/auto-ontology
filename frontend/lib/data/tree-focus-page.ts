// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SinglePageFormat } from '@/components/dataPage/SinglePageView';
import { ComposerSectionKind, DataModels, TreeFocusState } from '@/enums/datasources';
import type { Column, Database, Schema, Table } from '@/types/datasources';
import type { ComposerSection } from '@/types/composer-section';
import { isCatalogBranchLoadedForFocus } from '@/lib/data/catalog-branch-loaded';

export const WORKSPACE_ROOT_PARENT_ID = 'workspace-root';

export type TreeResolved =
	| { type: TreeFocusState.NONE }
	| { type: TreeFocusState.LOADING }
	| { type: DataModels.DB; database: Database }
	| { type: DataModels.SCHEMA; database: Database; schema: Schema }
	| { type: DataModels.TABLE; database: Database; schema: Schema; table: Table }
	| {
			type: DataModels.COLUMN;
			database: Database;
			schema: Schema;
			table: Table;
			column: Column;
	  };

function splitFocusSegments(focusId: string): string[] {
	return focusId.split('|').filter((segment) => segment.length > 0);
}

function resolvePendingColumnFocus(focusId: string, databases: Database[]): TreeResolved {
	const segments = splitFocusSegments(focusId);
	if (segments.length !== 4) return { type: TreeFocusState.NONE };
	const databaseId = segments[0];
	if (!databaseId || !databases.some((database) => database.id === databaseId))
		return { type: TreeFocusState.NONE };
	if (isCatalogBranchLoadedForFocus(databases, focusId)) return { type: TreeFocusState.NONE };
	return { type: TreeFocusState.LOADING };
}

export function resolveTreeNode(focusId: string | null, databases: Database[]): TreeResolved {
	if (!focusId) return { type: TreeFocusState.NONE };
	const segments = splitFocusSegments(focusId);
	if (segments.length === 0) return { type: TreeFocusState.NONE };

	for (const database of databases) {
		if (segments[0] !== database.id) continue;
		if (segments.length === 1) return { type: DataModels.DB, database };

		for (const schema of database.schemas) {
			if (segments[1] !== schema.id) continue;
			if (segments.length === 2) return { type: DataModels.SCHEMA, database, schema };

			for (const table of schema.tables) {
				if (segments[2] !== table.id) continue;
				if (segments.length === 3)
					return { type: DataModels.TABLE, database, schema, table };

				for (const column of table.columns) {
					if (segments.length === 4 && segments[3] === column.id) {
						return { type: DataModels.COLUMN, database, schema, table, column };
					}
				}
			}
		}
	}

	if (segments.length === 4 && databases.some((database) => database.id === segments[0])) {
		return resolvePendingColumnFocus(focusId, databases);
	}

	return { type: TreeFocusState.NONE };
}

function baseCardsForEntity(
	description: string,
	items: { label: string; value: string }[],
): ComposerSection[] {
	return [
		{
			type: ComposerSectionKind.TEXT_CARD,
			id: 'description',
			title: 'Description',
			body: description,
			editable: true,
		},
		{
			type: ComposerSectionKind.INFO_GRID,
			id: 'information',
			title: 'Information',
			items,
		},
	];
}

export function buildTreeFocusPageFormat(
	focusId: string | null,
	databases: Database[],
	workspaceDataId: string,
): SinglePageFormat {
	const resolvedFocus = resolveTreeNode(focusId, databases);

	if (resolvedFocus.type === TreeFocusState.LOADING) {
		const sections: ComposerSection[] = [
			{
				type: ComposerSectionKind.LOADING_PANEL,
				id: 'tree-focus-loading',
				message: 'Loading catalog details…',
			},
		];
		return {
			sections,
			header: {
				header: {
					title: 'Loading…',
					subtitle: 'Catalog',
				},
			},
		};
	}

	if (resolvedFocus.type === TreeFocusState.NONE) {
		const sections: ComposerSection[] = [
			{
				type: ComposerSectionKind.TEXT_CARD,
				id: 'catalog-intro',
				title: 'Catalog',
				body:
					databases.length === 0
						? 'No databases are available yet.'
						: 'Choose a database in the tree or pick one from the table below.',
			},
		];
		if (databases.length > 0) {
			sections.push({
				type: ComposerSectionKind.DATA_TABLE,
				id: 'all-databases',
				title: `Databases (${databases.length})`,
				columns: [
					{ key: 'name', label: 'Name' },
					{ key: 'schemas', label: 'Schemas' },
				],
				rows: databases.map((database) => ({
					name: database.name,
					schemas: String(database.num_of_schemas),
				})),
			});
		}
		return {
			sections,
			header: {
				header: {
					title: 'All Data',
					subtitle: 'Select a database, schema, table, or column in the tree.',
				},
			},
		};
	}

	const sections: ComposerSection[] = [];

	switch (resolvedFocus.type) {
		case DataModels.DB: {
			const { database } = resolvedFocus;
			sections.push(
				...baseCardsForEntity(database.description ?? '', [
					{ label: 'Schemas', value: String(database.schemas.length) },
				]),
			);
			sections.push({
				type: ComposerSectionKind.DATA_TABLE,
				id: 'child-schemas',
				title: `Schemas (${database.schemas.length})`,
				columns: [
					{ key: 'name', label: 'Name' },
					{ key: 'tables', label: 'Tables' },
				],
				rows: database.schemas.map((schema) => ({
					name: schema.schema_name,
					tables: String(schema.tables_count),
				})),
			});
			return {
				sections,
				header: {
					header: {
						title: database.name,
						subtitle: 'database',
						entityId: database.id,
						parentId: workspaceDataId,
					},
				},
			};
		}
		case DataModels.SCHEMA: {
			const { database, schema } = resolvedFocus;
			sections.push(
				...baseCardsForEntity(schema.description ?? '', [
					{ label: 'Schema', value: schema.schema_name },
					{ label: 'Database', value: database.name },
					{ label: 'Tables', value: String(schema.tables_count) },
				]),
			);
			sections.push({
				type: ComposerSectionKind.DATA_TABLE,
				id: 'child-tables',
				title: `Tables (${schema.tables.length})`,
				columns: [
					{ key: 'name', label: 'Name' },
					{ key: 'columns', label: 'Columns' },
				],
				rows: schema.tables.map((table) => ({
					name: table.name,
					columns: String(table.columns_count),
				})),
			});
			return {
				sections,
				header: {
					header: {
						title: schema.schema_name,
						subtitle: `Schema · ${database.name}`,
						entityId: schema.id,
						parentId: database.id,
					},
				},
			};
		}
		case DataModels.TABLE: {
			const { database, schema, table } = resolvedFocus;
			sections.push(
				...baseCardsForEntity(table.description ?? '', [
					{ label: 'Table', value: table.name },
					{ label: 'Schema', value: table.schema_name },
					{ label: 'Database', value: table.database_name },
					{ label: 'Columns', value: String(table.columns_count) },
				]),
			);
			sections.push({
				type: ComposerSectionKind.DATA_TABLE,
				id: 'child-columns',
				title: `Columns (${table.columns.length})`,
				columns: [
					{ key: 'ordinal_position', label: '#' },
					{ key: 'name', label: 'Name' },
					{ key: 'data_type', label: 'Type' },
				],
				rows: table.columns.map((column) => ({
					ordinal_position: String(column.ordinal_position),
					name: column.column_name,
					data_type: column.data_type,
				})),
			});
			return {
				sections,
				header: {
					header: {
						title: table.name,
						subtitle: `Table · ${schema.schema_name} · ${database.name}`,
						entityId: table.id,
						parentId: schema.id,
					},
				},
			};
		}
		case DataModels.COLUMN: {
			const { table, column } = resolvedFocus;
			sections.push(
				{
					type: ComposerSectionKind.TEXT_CARD,
					id: 'description',
					title: 'Description',
					body: column.description ?? '',
					editable: true,
				},
				{
					type: ComposerSectionKind.TAG_LIST,
					id: 'sample_values',
					title: 'Sample Values',
					values: column.sample_values ?? [],
					editable: true,
				},
				{
					type: ComposerSectionKind.INFO_GRID,
					id: 'information',
					title: 'Information',
					items: [
						{ label: 'Column', value: column.column_name },
						{
							label: 'Data type',
							value: column.data_type.trim() ? column.data_type : '—',
						},
						{ label: 'Table', value: column.table_name },
						{ label: 'Position', value: String(column.ordinal_position) },
					],
				},
			);
			return {
				sections,
				header: {
					header: {
						title: column.column_name,
						subtitle: `Column · ${column.table_name}`,
						entityId: column.id,
						parentId: table.id,
					},
				},
			};
		}
	}
}
