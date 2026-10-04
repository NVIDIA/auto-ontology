// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SinglePageFormat } from '@/common/SinglePageView';
import { IconName } from '@/common/icons';
import { catalogNodeInfo } from '@/components/dataPage/catalog-node-utils';
import {
	ComposerColumnType,
	ComposerSectionKind,
	DataModels,
	TreeFocusState,
} from '@/enums/datasources';
import type { Column, Database, Schema, Table } from '@/types/datasources';
import type { ComposerCertification, ComposerSection } from '@/types/composer-section';
import type { TagChip } from '@/types/tags';
import { isCatalogBranchLoadedForFocus } from '@/lib/data/catalog-branch-loaded';
import { fieldStatus } from '@/lib/certification';
import { entityTagsSection } from '@/lib/tags';
import { sampleValuesEditable, sampleValuesReadOnlyHint } from '@/lib/column-types';

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
	certification?: ComposerCertification,
): ComposerSection[] {
	return [
		{
			type: ComposerSectionKind.TEXT_CARD,
			id: 'description',
			title: 'Description',
			body: description,
			editable: true,
			certification,
		},
		{
			type: ComposerSectionKind.INFO_GRID,
			id: 'information',
			title: 'Information',
			items,
		},
	];
}

/**
 * The page for whichever catalog node *focusId* names, built from the tree the
 * caller already holds.
 *
 * *tagOptions* is every tag that exists, which the picker on a table's or a
 * column's page offers; the other kinds carry no tags, so an empty list is the
 * right answer for a caller that knows the focus is one of those.
 */
export function buildTreeFocusPageFormat(
	focusId: string | null,
	databases: Database[],
	tagOptions: TagChip[] = [],
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
			header: { header: { title: 'Loading…' } },
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
			header: { header: { title: 'All Data', icon: IconName.Database } },
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
						icon: catalogNodeInfo[DataModels.DB].icon,
						entityId: database.id,
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
				rowIdKey: 'id',
				columns: [
					{ key: 'name', label: 'Name' },
					{ key: 'columns', label: 'Columns' },
					{
						key: 'certification',
						label: 'Certification',
						type: ComposerColumnType.CERTIFICATION,
						align: 'center',
					},
				],
				rows: schema.tables.map((table) => ({
					id: table.id,
					name: table.name,
					columns: String(table.columns_count),
					certification: fieldStatus(table.description_certified === true),
				})),
			});
			return {
				sections,
				header: {
					header: {
						title: schema.schema_name,
						icon: catalogNodeInfo[DataModels.SCHEMA].icon,
						entityId: schema.id,
					},
				},
			};
		}
		case DataModels.TABLE: {
			const { table } = resolvedFocus;
			const tableTypeLabel = catalogNodeInfo[table.table_type].title;
			sections.push(
				...baseCardsForEntity(
					table.description ?? '',
					[
						{ label: 'Type', value: tableTypeLabel },
						{ label: 'Name', value: table.name },
						{ label: 'Schema', value: table.schema_name },
						{ label: 'Database', value: table.database_name },
						{ label: 'Columns', value: String(table.columns_count) },
					],
					{ certified: table.description_certified === true },
				),
			);
			sections.push(entityTagsSection(table.tags, tagOptions));
			sections.push({
				type: ComposerSectionKind.DATA_TABLE,
				id: 'child-columns',
				title: `Columns (${table.columns.length})`,
				rowIdKey: 'id',
				columns: [
					{ key: 'ordinal_position', label: '#' },
					{ key: 'name', label: 'Name' },
					{ key: 'data_type', label: 'Type' },
					{
						key: 'certification',
						label: 'Certification',
						type: ComposerColumnType.CERTIFICATION,
						align: 'center',
					},
				],
				rows: table.columns.map((column) => ({
					id: column.id,
					ordinal_position: String(column.ordinal_position),
					name: column.column_name,
					data_type: column.data_type,
					certification: fieldStatus(column.description_certified === true),
				})),
			});
			return {
				sections,
				header: {
					header: {
						title: table.name,
						icon: catalogNodeInfo[table.table_type].icon,
						entityId: table.id,
					},
				},
			};
		}
		case DataModels.COLUMN: {
			const { column } = resolvedFocus;
			const samplesEditable = sampleValuesEditable(column.data_type);
			sections.push(
				{
					type: ComposerSectionKind.TEXT_CARD,
					id: 'description',
					title: 'Description',
					body: column.description ?? '',
					editable: true,
					certification: { certified: column.description_certified === true },
				},
				{
					type: ComposerSectionKind.TAG_LIST,
					id: 'sample_values',
					title: 'Sample Values',
					values: Array.isArray(column.sample_values) ? column.sample_values : [],
					editable: samplesEditable,
					hint: samplesEditable ? undefined : sampleValuesReadOnlyHint(column.data_type),
				},
				entityTagsSection(column.tags, tagOptions),
				{
					type: ComposerSectionKind.INFO_GRID,
					id: 'information',
					title: 'Information',
					items: [
						{ label: 'Column', value: column.column_name },
						{
							label: 'Data Type',
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
						icon: catalogNodeInfo[DataModels.COLUMN].icon,
						entityId: column.id,
					},
				},
			};
		}
	}
}
