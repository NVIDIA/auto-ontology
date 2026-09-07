// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TableType } from '@/enums/datasources';
import type { ExplorationLayer } from '@/enums/exploration';
import type { Term, TermZone } from '@/types/terms';

// The exploration graph does not surface certification, so those Term fields
// are intentionally omitted from the graph node shape.
export type ExplorationTermNode = Omit<
	Term,
	'name_certified' | 'description_certified' | 'certification'
> & {
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

/**
 * One FK column pair joining two tables in an `ExplorationLink`. Mirrors
 * `DataGraphForeignKeyDto` (see `buildDataGraph`) — the server already
 * guarantees a plain array-or-`null` shape here.
 */
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

	relationshipTypes: string[];
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
	relationship_count: number;
	zones: TermZone[];
};

export type RelatedExplorationNodeDto = {
	id: string;
	name: string;
	relationship_count: number;
	table_type?: string;
	database_id?: string;
	schema_id?: string;
};

export type ExplorationRelationshipsPageDto = {
	nodes: RelatedExplorationNodeDto[];
	total: number;
};

export type SemanticExplorationGraph = {
	nodes: SemanticGraphNodeDto[];
	links: Array<{ source: string; target: string; relationship_types: string[] }>;
};

/** One Term/Table/Column/ColumnAttribute node along an `ExplorationLinkPathHop` chain. */
export type ExplorationLinkPathNodeDto = {
	id: string;
	name: string | null;
	/** Matches `ExpansionEntityKind`/`NodeType` (`term`, `table`, `column`, `columnAttribute`). */
	type: string;
	/** Only ever set for a `table`/`column` kind node — see
	 * `_enrich_catalog_path_nodes` in `gsf/dal/attributes.py`. Lets the client
	 * expand either further, the same way any other Table/Column node's
	 * expansion does. */
	database_id?: string | null;
	database_name?: string | null;
	schema_id?: string | null;
	schema_name?: string | null;
	/** Only ever set for a `column` kind node — its own owning Table. */
	table_id?: string | null;
	table_name?: string | null;
};

/** One relationship traversed along a term↔term path. */
export type ExplorationLinkPathHopDto = {
	relationship: string;
	source: ExplorationLinkPathNodeDto;
	target: ExplorationLinkPathNodeDto;
};

/**
 * Server DTO for the real ordered hop chain connecting two Terms — e.g.
 * Term1 <-REPRESENTS- Table -CONTAINS-> Column -SEMANTIC_FK-> ColumnAttribute
 * -PROPERTY_OF-> Term2 — returned when a term↔term Exploration graph edge is
 * clicked, so the client can graft/highlight the real path instead of just
 * the edge's own collapsed `relationship_types` label. `hops` is ordered
 * from the clicked edge's source term to its target term; empty only when
 * the two terms somehow share no path (shouldn't happen for a real edge).
 */
export type ExplorationLinkPathDto = {
	hops: ExplorationLinkPathHopDto[];
};

/**
 * Server DTO for one FK column pair joining two tables in a
 * `DataGraphEdgeDto` (see `ForeignKeyRef` in `gsf/server/models.py`). The
 * server renders both through `stringify_sample_values` before responding, so
 * these always land here as a plain string array (or `null`).
 */
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
	relationship_types: string[];
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
		/**
		 * The relationship kind(s) connecting this term to the
		 * table (e.g. `['REPRESENTS']`, `['HAS_ATTRIBUTE', 'SEMANTIC_FK']`) —
		 * shown as a label on the expansion edge, mirroring
		 * `ExplorationLink.relationshipTypes`.
		 */
		relationship_types: string[];
	}>;

	terms_total: number;
};

/** Server DTO for one page of Tables linked to a Term — the reverse of `TableExplorationDetails`. */
export type TermExplorationDetails = {
	tables: Array<{
		id: string;
		name: string;
		table_type: string | null;
		database_id: string | null;
		database_name: string | null;
		schema_id: string | null;
		schema_name: string | null;
		/**
		 * The relationship kind(s) connecting this table to
		 * the term — mirrors `TableExplorationDetails.terms[].relationship_types`,
		 * just walked from the other end.
		 */
		relationship_types: string[];
	}>;

	tables_total: number;
};

/**
 * Server DTO for a Column's own ColumnAttribute, outgoing FOREIGN_KEY
 * Column, incoming FOREIGN_KEY Columns, and referencing Sql queries.
 * `column_attribute`/`foreign_key_column` are `null` for most columns (a
 * column can have either, both, or neither); `referencing_columns` is the
 * reverse of `foreign_key_column` — every visible Column whose own FK
 * points *at* this one (typically this column is a table's primary key) —
 * empty when none do; `sql_queries` is every visible `Sql` node with a
 * direct `SQL` edge into this column — the reverse of
 * `SqlExplorationDetails.columns` — empty when no stored query ever
 * referenced it directly.
 */
export type ColumnExplorationDetails = {
	column_attribute: {
		id: string;
		name: string | null;
		description: string | null;
		/** The relationship kind — `HAS_ATTRIBUTE` or `SEMANTIC_FK`. */
		relationship_type: string | null;
	} | null;
	foreign_key_column: {
		id: string;
		name: string | null;
		description: string | null;
		data_type: string | null;
		table_id: string | null;
		table_name: string | null;
		database_id: string | null;
		database_name: string | null;
		schema_id: string | null;
		schema_name: string | null;
	} | null;
	referencing_columns: Array<{
		id: string;
		name: string | null;
		description: string | null;
		data_type: string | null;
		table_id: string | null;
		table_name: string | null;
		database_id: string | null;
		database_name: string | null;
		schema_id: string | null;
		schema_name: string | null;
	}>;
	sql_queries: Array<{
		id: string;
		sql: string;
	}>;
};

/**
 * Server DTO for a ColumnAttribute's own owning Term and every linked
 * Column — the reverse of `ColumnExplorationDetails` above, one hop
 * further out (`ColumnAttribute` rather than `Column`). `term` is `null`
 * when the attribute has no owning Term, or that Term is out of scope.
 * `columns` is one ordered page of *every* Column HAS_ATTRIBUTE/
 * SEMANTIC_FK-linked to this attribute — not just the "primary" one
 * whichever expansion grafted the attribute on already knew about.
 */
export type ColumnAttributeExplorationDetails = {
	term: { id: string; name: string | null; description: string | null } | null;
	columns: Array<{
		id: string;
		name: string | null;
		description: string | null;
		data_type: string | null;
		table_id: string | null;
		table_name: string | null;
		database_id: string | null;
		database_name: string | null;
		schema_id: string | null;
		schema_name: string | null;
		relationship_types: string[];
	}>;
	columns_total: number;
};

/**
 * Server DTO for a SqlAttribute's own Sql query and owning Term — the
 * reverse of `ColumnExplorationDetails` above, one hop further out
 * (`SqlAttribute` rather than `Column`). `sql` is `null` when the
 * attribute's SQL touches a table outside the caller's zones, `term` when
 * it somehow has no owning Term.
 */
export type SqlAttributeExplorationDetails = {
	sql: { id: string; sql: string | null } | null;
	term: { id: string; name: string | null; description: string | null } | null;
};

/**
 * Server DTO for the CustomAnalysis, Column and SqlAttribute nodes hanging
 * off a SqlAttribute's own Sql node. `custom_analyses` shares a node purely
 * by SQL text (see `fetch_sql_exploration_details`), so a CustomAnalysis
 * saved with the exact same SQL shares that node instead of getting its
 * own — empty for the common case of no CustomAnalysis sharing it.
 * `columns` is every visible Column the Sql node's own `SQL` edges reach
 * directly (the same real relationship the ingestion pipeline draws for
 * every column a parsed query references), enriched with each column's
 * owning Table/Schema/Database ids and names so it can be grafted onto the
 * graph as a fully expandable Column node. `tables` is every visible Table
 * the Sql node's own `SQL` edges reach directly — one hop out from the
 * statement — enriched with each table's owning
 * Schema/Database ids and names so it can be grafted on as a fully
 * expandable Table node, same as `expandTermNode`'s own Table neighbours.
 * `sql_attributes` is every visible SqlAttribute that backs this same Sql
 * node (the reverse of `SqlAttributeExplorationDetails.sql`) — see
 * `expandSqlNode` in `ExplorationView.tsx`.
 */
export type SqlExplorationDetails = {
	custom_analyses: Array<{
		id: string;
		name: string | null;
		description: string | null;
	}>;
	columns: Array<{
		id: string;
		name: string | null;
		description: string | null;
		data_type: string | null;
		table_id: string | null;
		table_name: string | null;
		database_id: string | null;
		database_name: string | null;
		schema_id: string | null;
		schema_name: string | null;
	}>;
	tables: Array<{
		id: string;
		name: string | null;
		table_type: string | null;
		database_id: string | null;
		database_name: string | null;
		schema_id: string | null;
		schema_name: string | null;
	}>;
	sql_attributes: Array<{
		id: string;
		name: string | null;
		description: string | null;
		term_id: string | null;
		term_name: string | null;
	}>;
};

/**
 * A Schema/Column/Term/Table/ColumnAttribute/SqlAttribute/Sql node grafted
 * onto the Data-layer graph by expanding a Table, Term, or Schema node (see
 * `GraphController.addExpansion` in `GraphCanvas.tsx`). Kept separate from
 * `ExplorationNode` — which is the shape of the *base* graph fetched up
 * front — since these are looked up from a side map (`ExplorationView`'s
 * `expandedNodesById`) rather than `graph.nodes`. A `table` entity here only
 * ever describes one *not* already present in the base graph (e.g. dropped
 * by `MAX_EXPLORATION_GRAPH_NODES` truncation) — a term/schema expanding
 * onto a table already in `graph.nodes` is shown via `ActiveDataCard`
 * instead. `columnAttribute`/`sqlAttribute` entities are the
 * `ColumnAttribute`/`SqlAttribute` nodes a Term's own expansion grafts on
 * (its `PROPERTY_OF` neighbours), the former distinct from a raw catalog
 * `column`. `sql` is the raw `Sql` query node a SqlAttribute's own
 * expansion grafts on in turn (its `HAS_SQL` neighbour) — see
 * `expandSqlAttributeNode` in `ExplorationView.tsx`. `customAnalysis` is a
 * CustomAnalysis node a `sql` node's own expansion grafts on in turn, for
 * the (uncommon) case where a CustomAnalysis was saved with the exact same
 * SQL text and so shares that same Sql node — see `expandSqlNode`.
 *
 * `connection` is not a graph node at all — it stands in for a clicked
 * term↔term Semantic-layer edge itself (see `activeSemanticConnectionEntity`
 * in `ExplorationView.tsx`) — a clicked table↔table Data-layer edge no
 * longer shows anything at all. `relationshipTypes`/`connectionHops` below
 * are its only real payload.
 */
export type ExpansionEntityKind =
	| 'schema'
	| 'column'
	| 'term'
	| 'table'
	| 'columnAttribute'
	| 'sqlAttribute'
	| 'sql'
	| 'customAnalysis'
	| 'connection';

export type ExpansionEntity = {
	id: string;
	kind: ExpansionEntityKind;
	name: string;
	description: string | null;
	/** Deep link to the entity's full page (Data catalog or Terms). */
	viewHref: string;
	/**
	 * Raw catalog ids/names backing a `schema`/`table`/`column` entity's own
	 * database (and, for `table`/`column`, its schema) — carried through
	 * (rather than re-derived from `id`/`viewHref`) so
	 * `expandSchemaNode`/`expandTableNode`/`expandColumnNode` in
	 * `ExplorationView.tsx` can fetch/graft the rest of the catalog path
	 * without a table-scoped closure to read it from. On a `table` entity
	 * (a Term's/Schema's/Column's own Table neighbour — see
	 * `expandTermNode`/`expandSchemaNode`/`expandColumnNode`), the *name*
	 * half can be `undefined` even when the id half isn't — its origin
	 * (e.g. a ColumnAttribute's `primary_column`, which the backend only
	 * ever gives raw ids, never names) sometimes has no display name on
	 * record; `handleDoubleClickNode`'s `case 'table'` falls back to the id
	 * itself in that case so the table still stays expandable either way.
	 * `undefined` for every other kind.
	 */
	databaseId?: string;
	databaseName?: string;
	/** The schema's own catalog id (unprefixed, unlike `id` above). */
	schemaId?: string;
	/**
	 * The schema's own catalog name — see the `databaseName` doc comment
	 * above for when this can be `undefined` alongside a present
	 * `schemaId`. `undefined` for every kind but `table`.
	 */
	schemaName?: string;
	/**
	 * The owning Table's catalog id/name for a `column` entity (from either a
	 * Table's own expansion or a `columnAttribute` entity's — see
	 * `expandColumnNode`), or the primary Column's catalog id/name for a
	 * `columnAttribute` entity (its `HAS_ATTRIBUTE` owner — see
	 * `expandColumnAttributeNode`). `undefined` for every other kind.
	 */
	tableId?: string;
	tableName?: string;
	columnId?: string;
	columnName?: string;
	/**
	 * A `column` entity's own catalog data type (e.g. `uuid`, `text`) — shown
	 * on `ActiveExpansionCard` alongside its description, which alone often
	 * reads as "no info" for a plain column with neither a description nor a
	 * linked ColumnAttribute to expand into. `undefined` for every other kind,
	 * and for a `column` entity whose origin never had it on hand (a Term's/
	 * ColumnAttribute's own Column neighbour, or a link-path graft).
	 */
	dataType?: string;
	/**
	 * A `columnAttribute` entity's own relationship kind to
	 * its owning Column — `HAS_ATTRIBUTE` or `SEMANTIC_FK` (see
	 * `expandColumnNode`'s own `column_attribute.relationship_type`) —
	 * shown on `ActiveExpansionCard` so an FK-shaped attribute (like
	 * `user_id`'s) reads as such rather than looking like a plain one.
	 * `undefined` for every other kind, and for a `columnAttribute` entity
	 * whose origin never had it on hand (a Term's own expansion — see
	 * `expandTermNode` — only ever carries the attribute's name/description).
	 */
	relationshipType?: string;
	/**
	 * The raw SQL text behind a `sql` entity (its own `Sql` node's
	 * `sql_full_query`) — see `expandSqlAttributeNode`. `undefined` for
	 * every other kind.
	 */
	sqlText?: string;
	/**
	 * A `connection` entity's own `ExplorationLink.relationshipTypes` — the
	 * relationship kind(s) (`REPRESENTS`, `HAS_ATTRIBUTE`,
	 * `SEMANTIC_FK`) reaching the two terms' shared table, same as drawn on
	 * the Semantic-layer edge itself. `undefined` for every other kind.
	 */
	relationshipTypes?: string[];
	/**
	 * A `connection` entity's own two Term endpoints — their raw ids (the
	 * halves `id` above joins with a `:`) paired with their display names,
	 * so `ActiveExpansionCard`'s "ID" row can read as `name: id, name: id`
	 * instead of two bare, unlabeled UUIDs. `undefined` for every other
	 * kind.
	 */
	connectionSource?: { id: string; name: string };
	connectionTarget?: { id: string; name: string };
	/**
	 * A `connection` entity's own real hop chain, fetched from
	 * `explorationApi.getSemanticLinkPath` — `null` while that request is
	 * still in flight (distinct from `[]`, an edge whose two terms
	 * genuinely share no path). `undefined` for every other kind.
	 */
	connectionHops?: ExplorationLinkPathHopDto[] | null;
};
