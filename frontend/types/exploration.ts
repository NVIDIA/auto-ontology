// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TableType } from '@/enums/datasources';
import type { ExplorationLayer } from '@/enums/exploration';
import type { Term, TermZone } from '@/types/terms';

export type ExplorationTermNode = Term & {
	layer: ExplorationLayer.Semantic;
	nodeType: 'term';
	relationshipCount: number;
	columnAttributesCount: number;
	sqlAttributesCount: number;
};

export type ExplorationDataNode = {
	id: string;
	name: string;
	description: string | null;
	layer: ExplorationLayer.Data;
	nodeType: TableType;
	relationshipCount: number;
	databaseId: string;
	databaseName: string;
	schemaId: string;
	schemaName: string;
	columnsCount: number;
	sqlCount: number;
	termsCount: number;
	zones: TermZone[];
};

export type ExplorationNode = ExplorationTermNode | ExplorationDataNode;

/** One FK column pair joining two tables in an `ExplorationLink`. */
export type ExplorationForeignKey = {
	sourceColumn: string;
	targetColumn: string;
	sourceSampleValues: string[] | null;
	targetSampleValues: string[] | null;
};

export type ExplorationLink = {
	source: string;
	target: string;
	queries: string[];
	/** True when this pair of tables is also (or only) linked by a foreign key. */
	viaForeignKey?: boolean;
	/** FK column pairs joining the two tables; empty for SQL-only links. */
	foreignKeys?: ExplorationForeignKey[];
};

export type ExplorationGraph = {
	nodes: ExplorationNode[];
	links: ExplorationLink[];
};

/** Maps a Table or Term id to the Zones it belongs to. */
export type ExplorationZonesMap = Record<string, TermZone[]>;

/** Server DTO for a semantic (Term) node in the Exploration graph endpoint. */
export type SemanticGraphNodeDto = {
	id: string;
	name: string;
	description: string | null;
	synonyms: string[];
	zones: TermZone[];
	relationship_count: number;
	column_attributes_count: number;
	sql_attributes_count: number;
};

/** Server DTO for a data (Table) node in the Exploration graph endpoint. */
export type DataGraphNodeDto = {
	id: string;
	name: string;
	description: string | null;
	table_type: string;
	database_id: string;
	database_name: string;
	schema_id: string;
	schema_name: string;
	columns_count: number;
	sql_count: number;
	terms_count: number;
	zones: TermZone[];
};

export type SemanticExplorationGraph = {
	nodes: SemanticGraphNodeDto[];
	links: Array<{ source: string; target: string }>;
};

/** Server DTO for one FK column pair joining two tables in a `DataGraphEdgeDto`. */
export type DataGraphForeignKeyDto = {
	source_column: string;
	target_column: string;
	source_sample_values: string[] | null;
	target_sample_values: string[] | null;
};

/** Server DTO for a data-layer Exploration edge (Table ↔ Table). */
export type DataGraphEdgeDto = {
	source: string;
	target: string;
	queries: string[];
	via_foreign_key: boolean;
	foreign_keys: DataGraphForeignKeyDto[];
};

export type DataExplorationGraph = {
	nodes: DataGraphNodeDto[];
	links: DataGraphEdgeDto[];
};

export type TableExplorationDetails = {
	queries: Array<{
		id: string;
		sql: string;
	}>;
	terms: Array<{
		id: string;
		name: string;
		description: string | null;
	}>;
};
