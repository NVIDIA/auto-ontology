// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import dynamic from 'next/dynamic';

import { datasources } from '@/api/datasources';
import { explorationApi } from '@/api/exploration';
import { termsApi } from '@/api/terms';
import { catalogNodeInfo } from '@/components/dataPage/catalog-node-utils';
import { SelectButton } from '@/common/Button';
import { EmptyState } from '@/common/EmptyState';
import { SelectButtonTheme } from '@/enums/button';
import { EmptyStateVariant } from '@/enums/emptyState';
import { Icon, IconName } from '@/common/icons';
import { catalogPathFromFocusId } from '@/lib/data/data-catalog-path';
import { SearchInput } from '@/common/SearchInput';
import { Text } from '@/common/Text';
import {
	ColumnAttributesModal,
	DataDetailsModal,
	RelationshipsModal,
	SqlAttributesModal,
	type DataDetailsType,
} from '@/common/modal';
import { TableType } from '@/enums/datasources';
import { ExplorationLayer } from '@/enums/exploration';
import type {
	ExpansionEntity,
	ExplorationDataNode,
	ExplorationGraph,
	ExplorationLinkPathHopDto,
	ExplorationTermNode,
} from '@/types/exploration';
import type {
	ExpansionEdgeInput,
	ExpansionNodeInput,
	GraphController,
	HighlightedPath,
	HoveredNode,
} from './graph/GraphCanvas';
import { NODE_TYPE_ACCENT_COLOR } from './graph/nodeTypeColors';
import type { NodeType } from './graph/nodeTypeColors';
import { ViewToggle } from './graph/ViewToggle';
import { ZoomControls } from './graph/ZoomControls';
import { ActiveExpansionCard } from './ActiveExpansionCard';
import { HoverNodeCard } from './HoverNodeCard';
import { ActiveDataCard, buildDataGraph } from './ExplorationData';
import { ExplorationLoader } from './ExplorationLoader';
import { ActiveTermCard, buildSemanticGraph } from './ExplorationSemantic';

const EMPTY_GRAPH: ExplorationGraph = { nodes: [], links: [] };

// Prefix for the "origin" id under which a clicked semantic edge's
// transient Table/Column/ColumnAttribute hop nodes are grafted onto the
// graph (see `expandSemanticConnection`) — one distinct origin per edge
// (rather than a single shared one) so expanding a connection behaves
// exactly like expanding a node: multiple connections can be expanded at
// once, and expanding/collapsing one never touches another's own graft
// (`GraphController` ref-counts nodes by origin — see `collapseSemanticConnection`).
// Never a real node id itself, so it can't collide with one.
const LINK_PATH_ORIGIN_PREFIX = '__link_path__:';
const linkPathOriginId = (edgeId: string): string => `${LINK_PATH_ORIGIN_PREFIX}${edgeId}`;

// The four types an `ExplorationLinkPathHopDto` node can be (see
// `find_term_link_path` in `auto_ontology/dal/attributes.py`) — narrows the backend's
// plain `string` `type` down to `ExpansionNodeInput`'s stricter `NodeType`
// union, since a hop node otherwise carries no compile-time guarantee of
// matching it.
const PATH_NODE_TYPES = ['term', 'table', 'column', 'columnAttribute'] as const;
type PathNodeType = (typeof PATH_NODE_TYPES)[number];
const isPathNodeType = (type: string): type is PathNodeType =>
	(PATH_NODE_TYPES as readonly string[]).includes(type);

/**
 * The node id a path hop's Table/Column/ColumnAttribute actually has (or
 * would have) on the live graph — Table/Term reuse their own real id
 * (`expandTermNode`'s own Table nodes, or the base graph's Term nodes,
 * already do), but Column/ColumnAttribute nodes are always prefixed (see
 * `expandTableNode`/`expandColumnAttributeNode`) to avoid colliding with a
 * Table/Term id from a completely different entity.
 */
const pathNodeGraphId = (node: { id: string; type: string }): string => {
	if (node.type === 'column') return `column:${node.id}`;
	if (node.type === 'columnAttribute') return `attribute:${node.id}`;
	return node.id;
};

// Caps on how many Column/Term nodes a single table's expansion grafts onto
// the graph — a very wide or heavily-tagged table would otherwise flood the
// canvas. The full lists stay one click away via the table's own side panel
// ("View columns"/"Terms" links open `DataDetailsModal`).
const COLUMN_EXPAND_LIMIT = 40;
const TERM_EXPAND_LIMIT = 20;
// Caps how many Sql query nodes a Table's own expansion grafts on — the
// same real `Sql-[SQL]->Table` edge `expandSqlNode` reads from the other
// end, capped for the same flood-avoidance reason as the two caps above: a
// heavily-queried table can have far more stored queries than are useful
// to draw on the canvas at once. The full list stays one click away via
// the table's own "View SQL queries" side panel link (`DataDetailsModal`).
const TABLE_SQL_EXPAND_LIMIT = 20;
// Same rationale as the two caps above, applied to a Term's own expansion —
// the Tables it links to (see `expandTermNode`).
const TERM_TABLE_EXPAND_LIMIT = 20;
// Caps how many of a Term's *related Terms* (see `expandTermNode`) get
// grafted on alongside its Tables — same rationale as the caps above, for a
// heavily cross-referenced term.
const TERM_RELATED_TERMS_EXPAND_LIMIT = 20;
// Caps how many of a Term's own ColumnAttribute nodes (its `PROPERTY_OF`
// neighbours — see `expandTermNode`) get grafted on alongside its Tables
// and related Terms, same rationale as the caps above.
const TERM_COLUMN_ATTRIBUTE_EXPAND_LIMIT = 40;
// Same as `TERM_COLUMN_ATTRIBUTE_EXPAND_LIMIT` above, for a Term's SqlAttribute
// nodes (also `PROPERTY_OF` it, just computed rather than Column-backed).
const TERM_SQL_ATTRIBUTE_EXPAND_LIMIT = 40;
// Caps how many of a Schema's own Tables (see `expandSchemaNode`) get
// grafted on — `datasources.getTablesForSchema` has no pagination of its
// own (unlike the caps above, which trim an already-paged request), so this
// slices the full response instead, same rationale as the other caps.
const SCHEMA_TABLE_EXPAND_LIMIT = 80;
// Caps how many Columns a ColumnAttribute's own expansion grafts on (see
// `expandColumnAttributeNode`) — a widely-shared attribute (e.g. a common
// `user_id`-shaped one) can otherwise be `HAS_ATTRIBUTE`/`SEMANTIC_FK`-linked
// from dozens of Columns across just as many Tables, same flood-avoidance
// rationale as the caps above.
const COLUMN_ATTRIBUTE_COLUMN_EXPAND_LIMIT = 40;
// Caps how many incoming-FOREIGN_KEY Column nodes a Column's own expansion
// grafts on (see `expandColumnNode`'s `referencing_columns` handling) — the
// reverse of a Column's own outgoing FK, same flood-avoidance rationale as
// the caps above: a shared lookup-style column (e.g. a table's `id`
// primary key) can be referenced by many more Columns than are useful to
// draw at once. The backend itself already caps this at 50 (see
// `fetch_column_exploration_details`); this trims that further for the
// canvas specifically.
const COLUMN_REFERENCING_EXPAND_LIMIT = 20;

/**
 * One entry in the bottom-right "Viewing: ..." legend — a dot/icon pair
 * tinted with `kind`'s own `NODE_TYPE_ACCENT_COLOR` (the exact same accent
 * `GraphCanvas.tsx` draws that kind's node border/icon with, so the legend
 * never drifts out of sync with what's actually on the canvas), plus its
 * label and, for the handful of kinds counted from the base graph payload
 * (`term`/`table`), a live count. The Semantic layer's own `columnAttribute`/
 * `sqlAttribute` and the Data layer's own `column`/`sql` are expansion-only
 * kinds (`GraphController.addExpansion`) with no fixed total to show, so
 * they're rendered as a plain color key instead (`count` left `undefined`).
 * Colors are inline `style`, not Tailwind's `text-[...]`/`bg-[...]`
 * arbitrary-value classes, since `NODE_TYPE_ACCENT_COLOR`'s values aren't
 * static string literals Tailwind's build-time scanner can pick up.
 */
const LegendItem = ({
	kind,
	icon,
	label,
	count,
}: {
	kind: NodeType;
	icon: IconName;
	label: string;
	count?: number;
}) => {
	const color = NODE_TYPE_ACCENT_COLOR[kind];
	return (
		<span className="flex items-center gap-1.5 text-xs font-medium text-body dark:text-zinc-200">
			<span
				className="h-3 w-3 rounded-full border"
				style={{ borderColor: color, backgroundColor: `${color}33` }}
			/>
			<Icon name={icon} className="h-4 w-4" style={{ color }} />
			{label}
			{count != null ? ` ${count}` : null}
		</span>
	);
};

// Sigma.js touches WebGL globals at module scope, so it can only be
// evaluated in the browser; loading it via `next/dynamic` with `ssr: false`
// keeps Next.js from crashing while server-rendering this client component.
const GraphCanvas = dynamic(
	() => import('./graph/GraphCanvas').then((graphCanvasModule) => graphCanvasModule.GraphCanvas),
	{ ssr: false },
);

export const ExplorationView = () => {
	const router = useRouter();
	const searchParams = useSearchParams();
	const semanticId = searchParams.get('semanticId');
	const dataId = searchParams.get('dataId');
	const layer: ExplorationLayer =
		dataId != null || searchParams.get('view') === 'data'
			? ExplorationLayer.Data
			: ExplorationLayer.Semantic;
	const activeNodeIdFromUrl = layer === ExplorationLayer.Semantic ? semanticId : dataId;
	const [semanticGraph, setSemanticGraph] = useState<ExplorationGraph>(EMPTY_GRAPH);
	const [dataGraph, setDataGraph] = useState<ExplorationGraph>(EMPTY_GRAPH);
	const [semanticLoading, setSemanticLoading] = useState(true);
	const [dataLoading, setDataLoading] = useState(false);
	const [semanticError, setSemanticError] = useState<string | null>(null);
	const [dataError, setDataError] = useState<string | null>(null);
	const [dataLoaded, setDataLoaded] = useState(false);
	const [search, setSearch] = useState('');
	const [activeNodeId, setActiveNodeId] = useState<string | null>(activeNodeIdFromUrl);

	const activeNodeIdRef = useRef<string | null>(activeNodeId);
	useEffect(() => {
		activeNodeIdRef.current = activeNodeId;
	}, [activeNodeId]);
	const [hoveredNodePosition, setHoveredNodePosition] = useState<HoveredNode | null>(null);
	const [selectedSemanticEdgeId, setSelectedSemanticEdgeId] = useState<string | null>(null);
	// The real hop chain behind each currently-expanded connection (see
	// `expandSemanticConnection`), keyed by edge id — fetched from
	// `explorationApi.getSemanticLinkPath`. A missing key means that edge's
	// request is still in flight (`ConnectionDetails` in
	// `ActiveExpansionCard.tsx` treats `undefined` the same as `null`,
	// distinct from `[]`, an edge whose two terms genuinely share no path),
	// read via `activeSemanticConnectionEntity`'s `connectionHops` below.
	// Kept as one map (rather than a single "current" value) so multiple
	// connections can be expanded at once, each remembering its own hops —
	// same rationale as `expandedNodesById` below for node expansions.
	const [connectionHopsByEdgeId, setConnectionHopsByEdgeId] = useState<
		Map<string, ExplorationLinkPathHopDto[]>
	>(new Map());
	// Mirrors `selectedSemanticEdgeId` for the same reason every other ref
	// in this file mirrors its state — `loadSemanticLinkPath`'s fetch can
	// resolve after the user has already clicked a different edge (or
	// deselected entirely), and this lets it recognize that and bail
	// instead of clobbering a newer selection's own path/highlight with a
	// stale response.
	const selectedSemanticEdgeIdRef = useRef<string | null>(null);
	useEffect(() => {
		selectedSemanticEdgeIdRef.current = selectedSemanticEdgeId;
	}, [selectedSemanticEdgeId]);
	// Which nodes/edges to highlight (dimming everything else) — set by
	// `handleSelectEdge` when a `relationship` edge on either layer is
	// clicked; see `HighlightedPath` in `GraphCanvas.tsx`.
	const [highlightedPath, setHighlightedPath] = useState<HighlightedPath | null>(null);
	const [relationshipsNodeId, setRelationshipsNodeId] = useState<string | null>(null);
	const [dataDetailsType, setDataDetailsType] = useState<DataDetailsType | null>(null);
	const [columnAttributesNodeId, setColumnAttributesNodeId] = useState<string | null>(null);
	const [sqlAttributesNodeId, setSqlAttributesNodeId] = useState<string | null>(null);
	const [controller, setController] = useState<GraphController | null>(null);
	// Mirrors `controller` (and, below, the expanded-table guard) in a ref
	// rather than reading the state value from `expandTableNode`'s closure.
	// `expandTableNode` must have a *stable* identity: it's a dependency of
	// `handleSelectNode`, which is passed to `GraphCanvas` as `onSelectNode` —
	// one of that component's main effect's dependencies. If expanding a
	// table (or the initial controller mount) changed `expandTableNode`'s
	// identity, it would tear down and rebuild the whole Sigma renderer,
	// which itself re-invokes `onControllerChange` with a new controller
	// object, changing identity again — an infinite render loop.
	const controllerRef = useRef<GraphController | null>(null);
	// Tables already grafted onto the live graph via `expandTableNode` — guards
	// against re-fetching/re-adding on a second click of the same table. A
	// ref (not state) for the same stability reason as `controllerRef` above;
	// nothing needs to re-render off this value changing.
	const expandedTableIdsRef = useRef<Set<string>>(new Set());
	// Mirrors `expandedTableIdsRef` above, for Term nodes grafted onto the
	// graph by a table's own expansion — see `expandTermNode`/`collapseTermNode`.
	const expandedTermIdsRef = useRef<Set<string>>(new Set());
	// Mirrors `expandedTermIdsRef` above, for Schema nodes grafted onto the
	// graph by a table's own expansion — see `expandSchemaNode`/`collapseSchemaNode`.
	const expandedSchemaIdsRef = useRef<Set<string>>(new Set());
	// Mirrors `expandedSchemaIdsRef` above, for ColumnAttribute nodes grafted
	// on by a term's own expansion — see `expandColumnAttributeNode`/
	// `collapseColumnAttributeNode`.
	const expandedColumnAttributeIdsRef = useRef<Set<string>>(new Set());
	// Mirrors `expandedColumnAttributeIdsRef` above, for Column nodes grafted
	// on by a table's or a ColumnAttribute's own expansion — see
	// `expandColumnNode`/`collapseColumnNode`.
	const expandedColumnIdsRef = useRef<Set<string>>(new Set());
	// Mirrors `expandedColumnAttributeIdsRef` above, for Sql nodes grafted on
	// by a SqlAttribute's own expansion — see
	// `expandSqlAttributeNode`/`collapseSqlAttributeNode`.
	const expandedSqlAttributeIdsRef = useRef<Set<string>>(new Set());
	// Mirrors `expandedSqlAttributeIdsRef` above, for CustomAnalysis nodes
	// grafted on by a Sql node's own expansion — see
	// `expandSqlNode`/`collapseSqlNode`.
	const expandedSqlIdsRef = useRef<Set<string>>(new Set());
	// Mirrors `expandedSqlIdsRef` above, for semantic edges (Term↔Term
	// "connections") whose real hop chain is currently grafted onto the
	// graph — see `expandSemanticConnection`/`collapseSemanticConnection`.
	const expandedConnectionIdsRef = useRef<Set<string>>(new Set());
	// Tables/terms/schemas/columnAttributes/columns/connections double-clicked to
	// collapse *while* their own expand fetch was still in flight —
	// `controller.removeExpansion` has nothing to remove yet in that case
	// (`addExpansion` hasn't run), so every `expandXNode` below checks this
	// set once its fetch resolves (or, for the synchronous ones, before
	// calling `addExpansion` at all) and skips grafting the nodes on,
	// instead of the collapse silently losing that race. Shared across
	// every kind: ids never collide across them.
	const pendingCollapseIdsRef = useRef<Set<string>>(new Set());
	// Name/description/id for every Schema/Column/Term/Table node an
	// expansion has added, keyed by id. Kept out of `dataGraph` so merging
	// into it never triggers `GraphCanvas`'s full rebuild effect (see
	// `GraphController`'s doc comment in `GraphCanvas.tsx`) — this map
	// exists purely so the side panel has something to show when one of
	// those nodes is clicked, and so `handleDoubleClickNode` can tell a
	// term's own expansion apart from a table's.
	const [expandedNodesById, setExpandedNodesById] = useState<Map<string, ExpansionEntity>>(
		new Map(),
	);
	// Mirrors `expandedNodesById` for the same reason `controllerRef` mirrors
	// `controller` (see its own comment above): `handleDoubleClickNode` reads
	// this to tell whether a double-clicked node is a term grafted on by a
	// prior expansion, without taking a dependency on the React-state map
	// itself — doing so would change `handleDoubleClickNode`'s identity (and
	// tear down/rebuild the whole Sigma renderer) on every single expand/collapse.
	const expandedNodesByIdRef = useRef<Map<string, ExpansionEntity>>(new Map());
	useEffect(() => {
		expandedNodesByIdRef.current = expandedNodesById;
	}, [expandedNodesById]);
	// Mirrors `semanticGraph` for the same stability reason as the ref above —
	// `expandTermNode` reads this to graft a Term's *related Terms* onto the
	// Data-layer graph alongside its linked Tables, the same neighbours that
	// term would show as graph edges on the Semantic layer itself (see
	// `fetch_semantic_exploration_graph`'s term↔term links), without taking a
	// dependency on the `semanticGraph` state value.
	const semanticGraphRef = useRef<ExplorationGraph>(EMPTY_GRAPH);
	useEffect(() => {
		semanticGraphRef.current = semanticGraph;
	}, [semanticGraph]);

	useEffect(() => {
		let cancelled = false;

		const loadGraph = async () => {
			setSemanticLoading(true);
			const response = await explorationApi.getSemanticExplorationGraph();
			if (cancelled) return;

			if (response.error) {
				setSemanticError(response.message ?? 'Failed to load terms');
				setSemanticLoading(false);
				return;
			}

			const nextGraph = buildSemanticGraph(response.data ?? { nodes: [], links: [] });
			if (cancelled) return;
			setSemanticGraph(nextGraph);
			setSemanticError(null);
			setSemanticLoading(false);
		};

		void loadGraph();
		return () => {
			cancelled = true;
		};
	}, []);

	useEffect(() => {
		if (layer !== ExplorationLayer.Data || dataLoaded) return undefined;
		let cancelled = false;

		const loadGraph = async () => {
			setDataLoading(true);
			try {
				const response = await explorationApi.getDataExplorationGraph();
				if (cancelled) return;
				if (response.error) {
					throw new Error(response.message ?? 'Failed to load data objects');
				}
				const nextGraph = buildDataGraph(response.data ?? { nodes: [], links: [] });
				if (cancelled) return;
				setDataGraph(nextGraph);
				setDataError(null);
				setDataLoaded(true);
			} catch (loadError) {
				if (cancelled) return;
				setDataError(
					loadError instanceof Error ? loadError.message : 'Failed to load data objects',
				);
			} finally {
				if (!cancelled) setDataLoading(false);
			}
		};

		void loadGraph();
		return () => {
			cancelled = true;
		};
	}, [dataLoaded, layer]);

	useEffect(() => {
		setActiveNodeId(activeNodeIdFromUrl);
	}, [activeNodeIdFromUrl]);

	// Every ref set above that tracks a kind of node/connection expansion,
	// together — the single list `handleToggleLayer` below clears in full
	// on every layer switch. Previously each ref was cleared there by a
	// separate hand-written line, and it's exactly that hand-maintained
	// list that silently fell out of sync with three of these refs; adding
	// a ref here is now the only thing a new expansion kind needs to be
	// included in that sweep. A plain `useRef` (rather than a fresh array
	// literal each render) so its identity is stable, same rationale as
	// every individual ref it holds.
	const expandedIdsRefs = useRef<RefObject<Set<string>>[]>([
		expandedTableIdsRef,
		expandedTermIdsRef,
		expandedSchemaIdsRef,
		expandedColumnAttributeIdsRef,
		expandedColumnIdsRef,
		expandedSqlAttributeIdsRef,
		expandedSqlIdsRef,
		expandedConnectionIdsRef,
	]);

	// Shared tail of every `collapseXNode`/`collapseSemanticConnection`
	// below: drop `trackingId` from that kind's own expansion-tracking ref
	// set, ask the controller to remove the graft, and — if it hasn't
	// grafted anything yet because its `expandXNode` fetch is still in
	// flight — flag it as pending instead so that fetch cancels itself
	// rather than grafting nodes on right after this collapse. `graphId`
	// defaults to `trackingId` for every caller except
	// `collapseSemanticConnection`, whose graft lives under a derived
	// origin id (see `linkPathOriginId`) rather than the raw edge id its
	// own ref set is keyed by. Returns whether the graft was actually
	// resolved (i.e. neither of the two guards above short-circuited it) —
	// `collapseSemanticConnection` uses this to decide whether it's safe to
	// drop this connection's own cached hops too.
	const collapseExpansion = useCallback(
		(expandedIdsRef: RefObject<Set<string>>, trackingId: string, graphId = trackingId) => {
			const activeController = controllerRef.current;
			if (activeController == null) return false;
			expandedIdsRef.current.delete(trackingId);
			const removedNodeIds = activeController.removeExpansion(graphId);
			if (removedNodeIds == null) {
				pendingCollapseIdsRef.current.add(graphId);
				return false;
			}
			if (removedNodeIds.length > 0) {
				setExpandedNodesById((previous) => {
					const next = new Map(previous);
					removedNodeIds.forEach((removedId) => next.delete(removedId));
					return next;
				});

				if (
					activeNodeIdRef.current != null &&
					removedNodeIds.includes(activeNodeIdRef.current)
				) {
					setActiveNodeId(null);
				}
			}
			return true;
		},
		[],
	);

	// Shared tail of every `expandXNode`/`expandSemanticConnection` below:
	// merges the entities it just fetched into `expandedNodesById` without
	// touching any entity some other expansion already put there.
	const mergeExpandedNodes = useCallback((newEntities: Map<string, ExpansionEntity>) => {
		setExpandedNodesById((previous) => {
			const next = new Map(previous);
			newEntities.forEach((value, key) => next.set(key, value));
			return next;
		});
	}, []);

	const isExpansionStale = useCallback(
		(pendingId: string, activeController: GraphController) =>
			pendingCollapseIdsRef.current.delete(pendingId) ||
			controllerRef.current !== activeController,
		[],
	);

	// Grafts a double-clicked table's Schema, Columns, linked Terms, and
	// referencing Sql queries onto the live graph — see
	// `GraphController.addExpansion` in `GraphCanvas.tsx`. Schema metadata
	// is already on the table node (no request needed); columns and
	// terms/queries come from two requests run in parallel. The Sql half is
	// the reverse of the same `Sql-[SQL]->Table` edge `expandSqlNode` draws
	// from the other end. Takes just the catalog fields it actually needs
	// (rather than a full `ExplorationDataNode`) so it also works for a
	// `table` entity grafted by a Term's own expansion (see
	// `expandTermNode`) — a Data-layer base graph node satisfies this shape
	// too, since it carries every field here plus more.
	const expandTableNode = useCallback(
		async (tableNode: {
			id: string;
			name: string;
			databaseId: string;
			databaseName: string;
			schemaId: string;
			schemaName: string;
		}) => {
			const activeController = controllerRef.current;
			if (activeController == null || expandedTableIdsRef.current.has(tableNode.id)) return;
			expandedTableIdsRef.current.add(tableNode.id);

			const schemaNodeId = `schema:${tableNode.schemaId}`;
			const expansionNodes: ExpansionNodeInput[] = [
				{ id: schemaNodeId, kind: 'schema', label: tableNode.schemaName },
			];
			const expansionEdges: ExpansionEdgeInput[] = [
				{ source: tableNode.id, target: schemaNodeId },
			];
			const newEntities = new Map<string, ExpansionEntity>();
			newEntities.set(schemaNodeId, {
				id: schemaNodeId,
				kind: 'schema',
				name: tableNode.schemaName,
				description: null,
				viewHref: catalogPathFromFocusId(`${tableNode.databaseId}|${tableNode.schemaId}`),
				databaseId: tableNode.databaseId,
				databaseName: tableNode.databaseName,
				schemaId: tableNode.schemaId,
			});

			const [columnsResponse, detailsResponse] = await Promise.all([
				datasources.getColumnsForTable(tableNode.id, {
					skip: 0,
					limit: COLUMN_EXPAND_LIMIT,
				}),
				explorationApi.getTableExplorationDetails(tableNode.id, {
					skip: 0,
					limit: TERM_EXPAND_LIMIT,
				}),
			]);

			if (!columnsResponse.error) {
				(columnsResponse.data ?? []).forEach((column) => {
					const columnNodeId = `column:${column.id}`;
					expansionNodes.push({
						id: columnNodeId,
						kind: 'column',
						label: column.column_name,
					});
					expansionEdges.push({ source: tableNode.id, target: columnNodeId });
					newEntities.set(columnNodeId, {
						id: columnNodeId,
						kind: 'column',
						name: column.column_name,
						description: column.description ?? null,
						viewHref: catalogPathFromFocusId(
							`${tableNode.databaseId}|${tableNode.schemaId}|${tableNode.id}|${column.id}`,
						),
						databaseId: tableNode.databaseId,
						// Carried through (unlike `expandColumnAttributeNode`'s own
						// column entity below, whose backing `AttributeColumnRef`
						// has no name fields at all) so `expandColumnNode` can
						// graft this column's owning Table back on with an
						// accurate Schema caption, not just its bare id.
						databaseName: tableNode.databaseName,
						schemaId: tableNode.schemaId,
						schemaName: tableNode.schemaName,
						tableId: tableNode.id,
						tableName: tableNode.name,
						dataType: column.data_type,
					});
				});
			}

			if (!detailsResponse.error && detailsResponse.data) {
				detailsResponse.data.terms.forEach((term) => {
					// Reuses the real term id as the node id (rather than a
					// prefixed one, unlike schema/column above), so a term
					// already linked from a previously-expanded table is
					// shared instead of duplicated.
					expansionNodes.push({ id: term.id, kind: 'term', label: term.name });
					expansionEdges.push({ source: tableNode.id, target: term.id });
					newEntities.set(term.id, {
						id: term.id,
						kind: 'term',
						name: term.name,
						description: term.description,
						viewHref: `/terms?focus=${encodeURIComponent(term.id)}`,
					});
				});

				// The reverse of `expandSqlNode`'s own `Sql-[SQL]->Table` graft —
				// every Sql query that referenced this table directly — the same
				// edge, walked from the table's end.
				// Reuses the `sql:${id}` node id `expandColumnNode`/
				// `expandSqlAttributeNode` give a Sql node, so one already on
				// the graph is shared instead of duplicated.
				detailsResponse.data.queries.slice(0, TABLE_SQL_EXPAND_LIMIT).forEach((query) => {
					const sqlNodeId = `sql:${query.id}`;
					expansionNodes.push({ id: sqlNodeId, kind: 'sql', label: 'SQL Query' });
					expansionEdges.push({ source: tableNode.id, target: sqlNodeId });
					newEntities.set(sqlNodeId, {
						id: sqlNodeId,
						kind: 'sql',
						name: 'SQL Query',
						description: null,
						viewHref: catalogPathFromFocusId(
							`${tableNode.databaseId}|${tableNode.schemaId}|${tableNode.id}`,
						),
						sqlText: query.sql,
					});
				});
			}

			if (isExpansionStale(tableNode.id, activeController)) return;

			activeController.addExpansion(tableNode.id, expansionNodes, expansionEdges);
			mergeExpandedNodes(newEntities);
		},
		[isExpansionStale, mergeExpandedNodes],
	);

	// Reverses `expandTableNode` on a double click of an already-expanded
	// table — see `GraphController.removeExpansion` in `GraphCanvas.tsx` for
	// how shared Schema/Term nodes (reachable from more than one expanded
	// table) are kept alive until every table referencing them is collapsed.
	// See `collapseExpansion` for the shared shape every `collapseXNode`
	// below follows too.
	const collapseTableNode = useCallback(
		(tableId: string) => collapseExpansion(expandedTableIdsRef, tableId),
		[collapseExpansion],
	);

	// Grafts a double-clicked Term's linked Tables, related Terms, and own
	// Column/SQL Attributes onto the live graph — the mirror image of
	// `expandTableNode` above. Used both for a Term node that a prior table
	// expansion grafted on (Data layer) and for a Term node double-clicked
	// directly on its own base graph (Semantic layer) — either way, only
	// `id` is needed to fetch and graft the rest. Related Terms are the
	// exact same neighbours that Term would show as graph edges over on the
	// Semantic layer (two Terms sharing at least one Table — see
	// `fetch_semantic_exploration_graph`), read from the already-loaded
	// `semanticGraphRef` rather than a separate request. Some of the Tables
	// may already be permanent nodes on the base graph (any table drawn
	// from `dataGraph` itself); `GraphController.addExpansion` tells the
	// two apart so collapsing this term later never removes one of those
	// (see its own doc comment in `GraphCanvas.tsx`).
	const expandTermNode = useCallback(
		async (termEntity: { id: string }) => {
			const activeController = controllerRef.current;
			if (activeController == null || expandedTermIdsRef.current.has(termEntity.id)) return;
			expandedTermIdsRef.current.add(termEntity.id);

			const [detailsResponse, columnAttributesResponse, sqlAttributesResponse] =
				await Promise.all([
					explorationApi.getTermExplorationDetails(termEntity.id, {
						skip: 0,
						limit: TERM_TABLE_EXPAND_LIMIT,
					}),
					termsApi.getColumnAttributes(termEntity.id, {
						skip: 0,
						limit: TERM_COLUMN_ATTRIBUTE_EXPAND_LIMIT,
					}),
					termsApi.getSqlAttributes(termEntity.id, {
						skip: 0,
						limit: TERM_SQL_ATTRIBUTE_EXPAND_LIMIT,
					}),
				]);

			const expansionNodes: ExpansionNodeInput[] = [];
			const expansionEdges: ExpansionEdgeInput[] = [];
			const newEntities = new Map<string, ExpansionEntity>();

			if (!detailsResponse.error && detailsResponse.data) {
				detailsResponse.data.tables.forEach((table) => {
					expansionNodes.push({ id: table.id, kind: 'table', label: table.name });
					expansionEdges.push({ source: termEntity.id, target: table.id });
					// Harmless to record even when `table.id` is already a
					// permanent node on the base graph — `activeNode` always
					// takes precedence over this map when a node is clicked
					// (see `activeExpansionEntity` below), so the entry is
					// simply unused in that case. The raw catalog fields below
					// (`undefined` when the table has none on record) let
					// `handleDoubleClickNode` expand this table in turn — see
					// `expandTableNode`.
					newEntities.set(table.id, {
						id: table.id,
						kind: 'table',
						name: table.name,
						description: null,
						viewHref: catalogPathFromFocusId(
							`${table.database_id}|${table.schema_id}|${table.id}`,
						),
						databaseId: table.database_id ?? undefined,
						databaseName: table.database_name ?? undefined,
						schemaId: table.schema_id ?? undefined,
						schemaName: table.schema_name ?? undefined,
					});
				});
			}

			const relatedTermIds = new Set<string>();
			semanticGraphRef.current.links.forEach((link) => {
				if (link.source !== termEntity.id && link.target !== termEntity.id) return;
				const relatedTermId = link.source === termEntity.id ? link.target : link.source;
				if (relatedTermId === termEntity.id || relatedTermIds.has(relatedTermId)) return;
				if (relatedTermIds.size >= TERM_RELATED_TERMS_EXPAND_LIMIT) return;
				const relatedTermNode = semanticGraphRef.current.nodes.find(
					(node) => node.id === relatedTermId,
				);
				if (relatedTermNode == null) return;
				relatedTermIds.add(relatedTermId);

				expansionNodes.push({
					id: relatedTermNode.id,
					kind: 'term',
					label: relatedTermNode.name,
				});
				expansionEdges.push({ source: termEntity.id, target: relatedTermNode.id });
				// `activeExpansionTermNode` (see below) always re-resolves a `term`
				// kind entity's full record straight from `semanticGraph` by id, so
				// this entry only needs to carry enough for `handleDoubleClickNode`
				// to recognize it as an expandable Term.
				newEntities.set(relatedTermNode.id, {
					id: relatedTermNode.id,
					kind: 'term',
					name: relatedTermNode.name,
					description: relatedTermNode.description,
					viewHref: `/terms?focus=${encodeURIComponent(relatedTermNode.id)}`,
				});
			});

			if (!columnAttributesResponse.error && columnAttributesResponse.data) {
				columnAttributesResponse.data.forEach((attribute) => {
					// Prefixed like `column`/`schema` above — a ColumnAttribute's own
					// id could otherwise collide with the id of the Column it's
					// backed by (see `primary_column` below).
					const attributeNodeId = `attribute:${attribute.id}`;
					expansionNodes.push({
						id: attributeNodeId,
						kind: 'columnAttribute',
						label: attribute.name,
					});
					// The real direction is `(ColumnAttribute)-[:PROPERTY_OF]->(Term)`
					// (see `fetch_column_attributes` in `auto_ontology/dal/terms.py`) — drawn
					// from the term regardless, same as every other structural edge
					// `addExpansion` grafts on.
					expansionEdges.push({ source: termEntity.id, target: attributeNodeId });
					newEntities.set(attributeNodeId, {
						id: attributeNodeId,
						kind: 'columnAttribute',
						name: attribute.name,
						description: attribute.description,
						viewHref:
							attribute.primary_column != null
								? catalogPathFromFocusId(
										`${attribute.primary_column.db_id}|${attribute.primary_column.schema_id}|${attribute.primary_column.table_id}|${attribute.primary_column.id}`,
									)
								: `/terms?focus=${encodeURIComponent(termEntity.id)}`,
						// Carried through so `expandColumnAttributeNode` can graft
						// this attribute's own `HAS_ATTRIBUTE` Column on without a
						// second request — `undefined` (and so unexpandable) when
						// no Column owns it.
						databaseId: attribute.primary_column?.db_id,
						schemaId: attribute.primary_column?.schema_id,
						tableId: attribute.primary_column?.table_id,
						tableName: attribute.primary_column?.table_name,
						columnId: attribute.primary_column?.id,
						columnName: attribute.primary_column?.column_name,
					});
				});
			}

			if (!sqlAttributesResponse.error && sqlAttributesResponse.data) {
				sqlAttributesResponse.data.forEach((attribute) => {
					// Prefixed like the ColumnAttribute nodes above — no catalog
					// column backs a SqlAttribute, so there's no id collision risk,
					// but the prefix keeps every expansion node kind consistent.
					const attributeNodeId = `attribute:${attribute.id}`;
					expansionNodes.push({
						id: attributeNodeId,
						kind: 'sqlAttribute',
						label: attribute.name,
					});
					expansionEdges.push({ source: termEntity.id, target: attributeNodeId });
					newEntities.set(attributeNodeId, {
						id: attributeNodeId,
						kind: 'sqlAttribute',
						name: attribute.name,
						description: attribute.description,
						// No catalog Column backs a SqlAttribute (it's computed, not
						// stored) — the only place to view it further is the Term.
						viewHref: `/terms?focus=${encodeURIComponent(termEntity.id)}`,
					});
				});
			}

			if (isExpansionStale(termEntity.id, activeController)) return;

			activeController.addExpansion(termEntity.id, expansionNodes, expansionEdges);
			mergeExpandedNodes(newEntities);
		},
		[isExpansionStale, mergeExpandedNodes],
	);

	// Reverses `expandTermNode` — see `collapseExpansion`'s shared shape.
	const collapseTermNode = useCallback(
		(termId: string) => collapseExpansion(expandedTermIdsRef, termId),
		[collapseExpansion],
	);

	// Grafts a double-clicked Schema's full table list onto the live graph —
	// a Schema node only ever exists as one already grafted on by
	// `expandTableNode`, so `schemaEntity.databaseId`/`schemaId` (carried on
	// it since its own creation) are always present here. Most of these
	// tables are typically already permanent nodes elsewhere on the base
	// graph (every table up to `MAX_EXPLORATION_GRAPH_NODES` loads up
	// front); `GraphController.addExpansion` still grafts the schema→table
	// edge onto those without duplicating the node itself (see its own doc
	// comment in `GraphCanvas.tsx`), so expanding a Schema mostly just draws
	// in the edges connecting it to tables that were already on the canvas.
	const expandSchemaNode = useCallback(
		async (schemaEntity: ExpansionEntity) => {
			const activeController = controllerRef.current;
			if (
				activeController == null ||
				schemaEntity.schemaId == null ||
				expandedSchemaIdsRef.current.has(schemaEntity.id)
			) {
				return;
			}
			expandedSchemaIdsRef.current.add(schemaEntity.id);

			const tablesResponse = await datasources.getTablesForSchema(schemaEntity.schemaId, {
				databaseName: schemaEntity.databaseName,
			});

			const expansionNodes: ExpansionNodeInput[] = [];
			const expansionEdges: ExpansionEdgeInput[] = [];
			const newEntities = new Map<string, ExpansionEntity>();

			if (!tablesResponse.error) {
				(tablesResponse.data ?? []).slice(0, SCHEMA_TABLE_EXPAND_LIMIT).forEach((table) => {
					// Reuses the real table id as the node id (like a table
					// expansion's own Term nodes), so a table already on the base
					// graph — or grafted on by another expansion — is shared
					// instead of duplicated.
					expansionNodes.push({ id: table.id, kind: 'table', label: table.name });
					expansionEdges.push({ source: schemaEntity.id, target: table.id });
					newEntities.set(table.id, {
						id: table.id,
						kind: 'table',
						name: table.name,
						description: table.description ?? null,
						viewHref: catalogPathFromFocusId(
							`${schemaEntity.databaseId}|${schemaEntity.schemaId}|${table.id}`,
						),
						// Carried through (same as a Term's own Table neighbours in
						// `expandTermNode`) so `handleDoubleClickNode` can expand
						// *this* table in turn via `expandTableNode` — without these,
						// its `databaseId`/`databaseName`/`schemaId`/`schemaName`
						// guard there silently no-ops on every table a Schema
						// expansion grafts on.
						databaseId: schemaEntity.databaseId,
						databaseName: schemaEntity.databaseName,
						schemaId: schemaEntity.schemaId,
						schemaName: schemaEntity.name,
					});
				});
			}

			if (isExpansionStale(schemaEntity.id, activeController)) return;

			activeController.addExpansion(schemaEntity.id, expansionNodes, expansionEdges);
			mergeExpandedNodes(newEntities);
		},
		[isExpansionStale, mergeExpandedNodes],
	);

	// Reverses `expandSchemaNode` — see `collapseExpansion`'s shared shape.
	const collapseSchemaNode = useCallback(
		(schemaNodeId: string) => collapseExpansion(expandedSchemaIdsRef, schemaNodeId),
		[collapseExpansion],
	);

	// Grafts a double-clicked ColumnAttribute's own owning Term *and* every
	// Column that shares it onto the live graph — the edges are
	// `(Column)-[:HAS_ATTRIBUTE|SEMANTIC_FK]->(ColumnAttribute)-[:PROPERTY_OF]->
	// (Term)`, both drawn from the attribute regardless of direction, same
	// convention as every other expansion edge. Unlike the Term half, the
	// Column half can't rely on whichever single Column the attribute was
	// already carrying (`columnId`/etc. — set by `expandTermNode`/
	// `expandColumnNode`, whichever grafted this attribute on in the first
	// place): a shared attribute (e.g. a common `user_id`-shaped one) is
	// typically linked from many Columns across many Tables at once — so both
	// halves need the one request below. Reuses the same `column:${id}`/
	// real term id node ids a Table's/Term's own expansion would use for
	// the same Column/Term, so either is shared instead of duplicated.
	const expandColumnAttributeNode = useCallback(
		async (attributeEntity: ExpansionEntity) => {
			const activeController = controllerRef.current;
			if (
				activeController == null ||
				expandedColumnAttributeIdsRef.current.has(attributeEntity.id)
			) {
				return;
			}
			expandedColumnAttributeIdsRef.current.add(attributeEntity.id);

			const expansionNodes: ExpansionNodeInput[] = [];
			const expansionEdges: ExpansionEdgeInput[] = [];
			const newEntities = new Map<string, ExpansionEntity>();

			// Strips the `attribute:` prefix `expandTermNode`/`expandColumnNode`
			// give this node's id back down to the raw ColumnAttribute id the
			// details endpoint expects.
			const rawAttrId = attributeEntity.id.replace(/^attribute:/, '');
			const detailsResponse = await explorationApi.getColumnAttributeExplorationDetails(
				rawAttrId,
				{ skip: 0, limit: COLUMN_ATTRIBUTE_COLUMN_EXPAND_LIMIT },
			);

			if (!detailsResponse.error) {
				(detailsResponse.data?.columns ?? []).forEach((column) => {
					const columnNodeId = `column:${column.id}`;
					expansionNodes.push({
						id: columnNodeId,
						kind: 'column',
						label: column.name ?? '',
					});
					expansionEdges.push({ source: attributeEntity.id, target: columnNodeId });
					newEntities.set(columnNodeId, {
						id: columnNodeId,
						kind: 'column',
						name: column.name ?? '',
						description: column.description,
						viewHref:
							column.database_id != null &&
							column.schema_id != null &&
							column.table_id != null
								? catalogPathFromFocusId(
										`${column.database_id}|${column.schema_id}|${column.table_id}|${column.id}`,
									)
								: '/data',
						// Carried through (same rationale as `expandTableNode`'s own
						// column entities) so double-clicking this column in turn can
						// graft its own owning Table back on via `expandColumnNode`.
						databaseId: column.database_id ?? undefined,
						databaseName: column.database_name ?? undefined,
						schemaId: column.schema_id ?? undefined,
						schemaName: column.schema_name ?? undefined,
						tableId: column.table_id ?? undefined,
						tableName: column.table_name ?? undefined,
						dataType: column.data_type ?? undefined,
					});
				});
			}

			const term = detailsResponse.error ? null : (detailsResponse.data?.term ?? null);
			if (term != null) {
				// Reuses the real term id as the node id, like every other Term
				// node an expansion grafts on — shared with the one already on
				// the graph instead of duplicated.
				expansionNodes.push({ id: term.id, kind: 'term', label: term.name ?? '' });
				expansionEdges.push({ source: attributeEntity.id, target: term.id });
				newEntities.set(term.id, {
					id: term.id,
					kind: 'term',
					name: term.name ?? '',
					description: term.description,
					viewHref: `/terms?focus=${encodeURIComponent(term.id)}`,
				});
			}

			if (isExpansionStale(attributeEntity.id, activeController)) return;

			activeController.addExpansion(attributeEntity.id, expansionNodes, expansionEdges);
			mergeExpandedNodes(newEntities);
		},
		[isExpansionStale, mergeExpandedNodes],
	);

	// Reverses `expandColumnAttributeNode` — see `collapseExpansion`'s
	// shared shape.
	const collapseColumnAttributeNode = useCallback(
		(attributeId: string) => collapseExpansion(expandedColumnAttributeIdsRef, attributeId),
		[collapseExpansion],
	);

	// Grafts a double-clicked Column's own owning Table, its own
	// ColumnAttribute (if it has one), its own outgoing *and* incoming
	// FOREIGN_KEY Columns, and any Sql query that references it directly,
	// onto the live graph. The Table half mirrors `expandColumnAttributeNode`
	// above, one `CONTAINS` hop further out, and needs no request:
	// `columnEntity.tableId`/etc. were already carried on it by whichever
	// expansion (Table's or ColumnAttribute's) grafted this Column on in
	// the first place. Reuses the Table's real id (like a Term/Schema
	// expansion's own Table nodes), so a Table already on the base graph —
	// almost always the case, either on the Data layer's own
	// `dataGraph.nodes` or already grafted on elsewhere — is shared instead
	// of duplicated. Its own `databaseId`/`schemaId` (and, when known,
	// `databaseName`/`schemaName`) are carried through the same way, so
	// double-clicking it afterward — on *either* layer — finds it either
	// already on the base graph (see `handleDoubleClickNode`'s
	// `dataGraph.nodes` lookup) or expands it via that carried-through entity
	// (see its `case 'table'` below) instead of silently doing nothing. The
	// ColumnAttribute half is the reverse of the same `HAS_ATTRIBUTE`/
	// `SEMANTIC_FK` edge `expandColumnAttributeNode` draws from the other
	// end, the outgoing FOREIGN_KEY half is the *physical* FK constraint
	// (see `fetch_data_exploration_edges`'s identical traversal), the
	// incoming FOREIGN_KEY half is that same edge's reverse (every Column
	// whose own FK points at this one — typically the other side of a
	// primary/foreign key pair), and the Sql half is the reverse of the
	// same `Sql-[SQL]->Column` edge `expandSqlNode` draws from the other
	// end — unlike the Table half, all four of these *do* need a request: a
	// Column doesn't already know whether some ColumnAttribute owns it,
	// which Column its own FK (if any) points at, which Columns point at
	// it in turn, or which Sql queries (if any) reference it.
	const expandColumnNode = useCallback(
		async (columnEntity: ExpansionEntity) => {
			const activeController = controllerRef.current;
			if (
				activeController == null ||
				columnEntity.tableId == null ||
				expandedColumnIdsRef.current.has(columnEntity.id)
			) {
				return;
			}
			expandedColumnIdsRef.current.add(columnEntity.id);

			const { tableId } = columnEntity;
			// Strips the `column:` prefix `expandTableNode`/`expandColumnAttributeNode`
			// give this node's id (see either's own comment) back down to the raw
			// catalog id the details endpoint expects.
			const rawColumnId = columnEntity.id.replace(/^column:/, '');
			const expansionNodes: ExpansionNodeInput[] = [
				{ id: tableId, kind: 'table', label: columnEntity.tableName ?? '' },
			];
			const expansionEdges: ExpansionEdgeInput[] = [
				{ source: columnEntity.id, target: tableId },
			];
			const newEntities = new Map<string, ExpansionEntity>();
			newEntities.set(tableId, {
				id: tableId,
				kind: 'table',
				name: columnEntity.tableName ?? '',
				description: null,
				viewHref:
					columnEntity.databaseId != null && columnEntity.schemaId != null
						? catalogPathFromFocusId(
								`${columnEntity.databaseId}|${columnEntity.schemaId}|${tableId}`,
							)
						: '/data',
				// Carried through (`undefined` when the owning Column's own origin
				// never had them — see `expandColumnAttributeNode`'s column entity
				// below) so `handleDoubleClickNode` can expand *this* table in turn
				// via `expandTableNode`, same as a Term's/Schema's own Table
				// neighbours.
				databaseId: columnEntity.databaseId,
				databaseName: columnEntity.databaseName,
				schemaId: columnEntity.schemaId,
				schemaName: columnEntity.schemaName,
			});

			const detailsResponse = await explorationApi.getColumnExplorationDetails(rawColumnId);
			const attribute = detailsResponse.error
				? null
				: (detailsResponse.data?.column_attribute ?? null);
			if (attribute != null) {
				// Reuses the same `attribute:${id}` node id a Term's own expansion
				// would use for this ColumnAttribute (see `expandTermNode`), so the
				// two share one node instead of duplicating it.
				const attributeNodeId = `attribute:${attribute.id}`;
				expansionNodes.push({
					id: attributeNodeId,
					kind: 'columnAttribute',
					label: attribute.name ?? '',
				});
				expansionEdges.push({ source: columnEntity.id, target: attributeNodeId });
				newEntities.set(attributeNodeId, {
					id: attributeNodeId,
					kind: 'columnAttribute',
					name: attribute.name ?? '',
					description: attribute.description,
					viewHref:
						columnEntity.databaseId != null && columnEntity.schemaId != null
							? catalogPathFromFocusId(
									`${columnEntity.databaseId}|${columnEntity.schemaId}|${tableId}|${rawColumnId}`,
								)
							: '/data',
					databaseId: columnEntity.databaseId,
					schemaId: columnEntity.schemaId,
					tableId,
					tableName: columnEntity.tableName,
					columnId: rawColumnId,
					columnName: columnEntity.name,
					relationshipType: attribute.relationship_type ?? undefined,
				});
			}

			const sqlQueries = detailsResponse.error
				? []
				: (detailsResponse.data?.sql_queries ?? []);
			sqlQueries.forEach((sql) => {
				// Reuses the `sql:${id}` node id a SqlAttribute's own expansion
				// would use for this Sql node (see `expandSqlAttributeNode`), so
				// the two share one node instead of duplicating it.
				const sqlNodeId = `sql:${sql.id}`;
				expansionNodes.push({ id: sqlNodeId, kind: 'sql', label: 'SQL Query' });
				expansionEdges.push({ source: columnEntity.id, target: sqlNodeId });
				newEntities.set(sqlNodeId, {
					id: sqlNodeId,
					kind: 'sql',
					name: 'SQL Query',
					description: null,
					viewHref: columnEntity.viewHref,
					sqlText: sql.sql,
				});
			});

			const foreignKeyColumn = detailsResponse.error
				? null
				: (detailsResponse.data?.foreign_key_column ?? null);
			if (foreignKeyColumn != null) {
				// Reuses the `column:${id}` node id `expandTableNode` gives every
				// Column node (see `pathNodeGraphId`'s own comment above), so a
				// target column already grafted on by some other expansion is
				// shared instead of duplicated.
				const fkColumnNodeId = `column:${foreignKeyColumn.id}`;
				expansionNodes.push({
					id: fkColumnNodeId,
					kind: 'column',
					label: foreignKeyColumn.name ?? '',
				});
				expansionEdges.push({ source: columnEntity.id, target: fkColumnNodeId });
				newEntities.set(fkColumnNodeId, {
					id: fkColumnNodeId,
					kind: 'column',
					name: foreignKeyColumn.name ?? '',
					description: foreignKeyColumn.description,
					viewHref:
						foreignKeyColumn.database_id != null &&
						foreignKeyColumn.schema_id != null &&
						foreignKeyColumn.table_id != null
							? catalogPathFromFocusId(
									`${foreignKeyColumn.database_id}|${foreignKeyColumn.schema_id}|${foreignKeyColumn.table_id}|${foreignKeyColumn.id}`,
								)
							: '/data',
					// Carried through (same rationale as `expandTableNode`'s own
					// column entities) so double-clicking this FK-target column in
					// turn can graft its own owning Table back on via
					// `expandColumnNode` above, or itself be expanded further.
					databaseId: foreignKeyColumn.database_id ?? undefined,
					databaseName: foreignKeyColumn.database_name ?? undefined,
					schemaId: foreignKeyColumn.schema_id ?? undefined,
					schemaName: foreignKeyColumn.schema_name ?? undefined,
					tableId: foreignKeyColumn.table_id ?? undefined,
					tableName: foreignKeyColumn.table_name ?? undefined,
					dataType: foreignKeyColumn.data_type ?? undefined,
				});
			}

			// The reverse of `foreignKeyColumn` above — every Column whose own
			// outgoing FK points *at* this one (typically this Column is a
			// table's primary key and the others are foreign keys into it).
			// Without this, expanding an FK target Column showed nothing for
			// the referencing side even though it's a real, direct edge.
			const referencingColumns = detailsResponse.error
				? []
				: (detailsResponse.data?.referencing_columns ?? []);
			referencingColumns.slice(0, COLUMN_REFERENCING_EXPAND_LIMIT).forEach((referencing) => {
				// Reuses the `column:${id}` node id `expandTableNode` gives
				// every Column node, so a referencing column already grafted on
				// by some other expansion is shared instead of duplicated.
				const referencingNodeId = `column:${referencing.id}`;
				expansionNodes.push({
					id: referencingNodeId,
					kind: 'column',
					label: referencing.name ?? '',
				});
				expansionEdges.push({ source: columnEntity.id, target: referencingNodeId });
				newEntities.set(referencingNodeId, {
					id: referencingNodeId,
					kind: 'column',
					name: referencing.name ?? '',
					description: referencing.description,
					viewHref:
						referencing.database_id != null &&
						referencing.schema_id != null &&
						referencing.table_id != null
							? catalogPathFromFocusId(
									`${referencing.database_id}|${referencing.schema_id}|${referencing.table_id}|${referencing.id}`,
								)
							: '/data',
					// Carried through (same rationale as `expandTableNode`'s own
					// column entities) so double-clicking this referencing
					// column in turn can graft its own owning Table back on via
					// `expandColumnNode`.
					databaseId: referencing.database_id ?? undefined,
					databaseName: referencing.database_name ?? undefined,
					schemaId: referencing.schema_id ?? undefined,
					schemaName: referencing.schema_name ?? undefined,
					tableId: referencing.table_id ?? undefined,
					tableName: referencing.table_name ?? undefined,
					dataType: referencing.data_type ?? undefined,
				});
			});

			if (isExpansionStale(columnEntity.id, activeController)) return;

			activeController.addExpansion(columnEntity.id, expansionNodes, expansionEdges);
			mergeExpandedNodes(newEntities);
		},
		[isExpansionStale, mergeExpandedNodes],
	);

	// Reverses `expandColumnNode` — see `collapseExpansion`'s shared shape.
	const collapseColumnNode = useCallback(
		(columnId: string) => collapseExpansion(expandedColumnIdsRef, columnId),
		[collapseExpansion],
	);

	// Grafts a double-clicked SqlAttribute's own owning Term and its own Sql
	// query node onto the live graph — the SqlAttribute counterpart to
	// `expandColumnAttributeNode` above, one more hop out on both ends: the
	// Term (`PROPERTY_OF`, reconnecting it in case it isn't already on the
	// graph — almost always is, since it's the very Term whose own
	// expansion grafted this SqlAttribute on) and the Sql query itself
	// (`HAS_SQL`), the raw text `ActiveExpansionCard` renders via
	// `SqlBlock`. Unlike `expandColumnAttributeNode`, neither is already
	// known on `attributeEntity` (see `expandTermNode`, which only carries
	// the SqlAttribute's own name/description) — one request fetches both.
	const expandSqlAttributeNode = useCallback(
		async (attributeEntity: ExpansionEntity) => {
			const activeController = controllerRef.current;
			if (
				activeController == null ||
				expandedSqlAttributeIdsRef.current.has(attributeEntity.id)
			) {
				return;
			}
			expandedSqlAttributeIdsRef.current.add(attributeEntity.id);

			// Strips the `attribute:` prefix `expandTermNode` gives this node's
			// id back down to the raw SqlAttribute id the details endpoint expects.
			const rawAttrId = attributeEntity.id.replace(/^attribute:/, '');
			const detailsResponse =
				await explorationApi.getSqlAttributeExplorationDetails(rawAttrId);
			const details = detailsResponse.error ? null : detailsResponse.data;

			const expansionNodes: ExpansionNodeInput[] = [];
			const expansionEdges: ExpansionEdgeInput[] = [];
			const newEntities = new Map<string, ExpansionEntity>();

			if (details?.term != null) {
				const { term } = details;
				// Reuses the real term id as the node id, like every other Term
				// node an expansion grafts on — shared with the one already on
				// the graph instead of duplicated.
				expansionNodes.push({ id: term.id, kind: 'term', label: term.name ?? '' });
				expansionEdges.push({ source: attributeEntity.id, target: term.id });
				newEntities.set(term.id, {
					id: term.id,
					kind: 'term',
					name: term.name ?? '',
					description: term.description,
					viewHref: `/terms?focus=${encodeURIComponent(term.id)}`,
				});
			}

			if (details?.sql != null) {
				const { sql } = details;
				// Prefixed like `column`/`attribute` above — a Sql node's own id
				// could otherwise collide with some other kind's.
				const sqlNodeId = `sql:${sql.id}`;
				expansionNodes.push({ id: sqlNodeId, kind: 'sql', label: 'SQL Query' });
				expansionEdges.push({ source: attributeEntity.id, target: sqlNodeId });
				newEntities.set(sqlNodeId, {
					id: sqlNodeId,
					kind: 'sql',
					name: 'SQL Query',
					description: null,
					viewHref: attributeEntity.viewHref,
					sqlText: sql.sql ?? '',
				});
			}

			if (isExpansionStale(attributeEntity.id, activeController)) return;

			activeController.addExpansion(attributeEntity.id, expansionNodes, expansionEdges);
			mergeExpandedNodes(newEntities);
		},
		[isExpansionStale, mergeExpandedNodes],
	);

	// Reverses `expandSqlAttributeNode` — see `collapseExpansion`'s shared
	// shape.
	const collapseSqlAttributeNode = useCallback(
		(attributeId: string) => collapseExpansion(expandedSqlAttributeIdsRef, attributeId),
		[collapseExpansion],
	);

	// Grafts a double-clicked Sql node's own CustomAnalysis neighbours,
	// referenced Columns and Tables, and backing SqlAttributes onto the live
	// graph. The CustomAnalysis half is the (uncommon) other end of the
	// sharing `expandSqlAttributeNode`'s own doc comment describes: a
	// CustomAnalysis saved with the exact same SQL text as this
	// SqlAttribute's query shares this very Sql node rather than getting
	// its own — most Sql nodes have none. The Column half is the same real
	// `Sql-[SQL]->Column` edge the ingestion pipeline draws for every
	// column a parsed query references, and the Table half is the
	// identical `Sql-[SQL]->Table` edge `fetch_data_exploration_edges`
	// already reads elsewhere — unlike the
	// CustomAnalysis half, most Sql nodes *do* have at least one of these,
	// so this is what actually makes a Sql node's expansion show something
	// in the common case. The SqlAttribute half is the reverse of the same
	// `HAS_SQL` edge `expandSqlAttributeNode` draws from the other end —
	// reconnecting the very SqlAttribute whose own expansion grafted this
	// Sql node on in the first place (already on the graph, so a no-op for
	// that one specifically — see `GraphController.addExpansion`'s own
	// de-dup) plus, unlike that single already-known one, every *other*
	// SqlAttribute that happens to share this exact SQL text too. Needs one
	// request, since none of the four is known ahead of time. Reuses the
	// `column:${id}`/`attribute:${id}`/real table id node ids a Table's/
	// ColumnAttribute's/Term's own expansion would use for the same
	// Column/SqlAttribute/Table, so any is shared instead of duplicated.
	const expandSqlNode = useCallback(
		async (sqlEntity: ExpansionEntity) => {
			const activeController = controllerRef.current;
			if (activeController == null || expandedSqlIdsRef.current.has(sqlEntity.id)) {
				return;
			}
			expandedSqlIdsRef.current.add(sqlEntity.id);

			// Strips the `sql:` prefix `expandSqlAttributeNode` gives this
			// node's id back down to the raw Sql id the details endpoint expects.
			const rawSqlId = sqlEntity.id.replace(/^sql:/, '');
			const detailsResponse = await explorationApi.getSqlExplorationDetails(rawSqlId);

			const expansionNodes: ExpansionNodeInput[] = [];
			const expansionEdges: ExpansionEdgeInput[] = [];
			const newEntities = new Map<string, ExpansionEntity>();

			if (!detailsResponse.error && detailsResponse.data) {
				detailsResponse.data.custom_analyses.forEach((analysis) => {
					// Prefixed like `attribute`/`column` above — a CustomAnalysis's
					// own id could otherwise collide with some other kind's.
					const analysisNodeId = `customAnalysis:${analysis.id}`;
					expansionNodes.push({
						id: analysisNodeId,
						kind: 'customAnalysis',
						label: analysis.name ?? '',
					});
					expansionEdges.push({ source: sqlEntity.id, target: analysisNodeId });
					newEntities.set(analysisNodeId, {
						id: analysisNodeId,
						kind: 'customAnalysis',
						name: analysis.name ?? '',
						description: analysis.description,
						// No per-analysis focus deep link exists (unlike
						// `/data`/`/terms`) — the list page is the closest
						// thing to a "view" for a CustomAnalysis.
						viewHref: '/analysis',
					});
				});

				detailsResponse.data.columns.forEach((column) => {
					const columnNodeId = `column:${column.id}`;
					expansionNodes.push({
						id: columnNodeId,
						kind: 'column',
						label: column.name ?? '',
					});
					expansionEdges.push({ source: sqlEntity.id, target: columnNodeId });
					newEntities.set(columnNodeId, {
						id: columnNodeId,
						kind: 'column',
						name: column.name ?? '',
						description: column.description,
						viewHref:
							column.database_id != null &&
							column.schema_id != null &&
							column.table_id != null
								? catalogPathFromFocusId(
										`${column.database_id}|${column.schema_id}|${column.table_id}|${column.id}`,
									)
								: '/data',
						// Carried through (same rationale as `expandTableNode`'s own
						// column entities) so double-clicking this column in turn
						// can graft its own owning Table back on via
						// `expandColumnNode`.
						databaseId: column.database_id ?? undefined,
						databaseName: column.database_name ?? undefined,
						schemaId: column.schema_id ?? undefined,
						schemaName: column.schema_name ?? undefined,
						tableId: column.table_id ?? undefined,
						tableName: column.table_name ?? undefined,
						dataType: column.data_type ?? undefined,
					});
				});

				detailsResponse.data.tables.forEach((table) => {
					// Reuses the real table id as the node id — like a Term's own
					// Table neighbours in `expandTermNode` — so a table already on
					// the base graph (almost always the case) is shared instead of
					// duplicated, mirroring the real `Sql-[SQL]->Table` edge.
					expansionNodes.push({ id: table.id, kind: 'table', label: table.name ?? '' });
					expansionEdges.push({ source: sqlEntity.id, target: table.id });
					newEntities.set(table.id, {
						id: table.id,
						kind: 'table',
						name: table.name ?? '',
						description: null,
						viewHref:
							table.database_id != null && table.schema_id != null
								? catalogPathFromFocusId(
										`${table.database_id}|${table.schema_id}|${table.id}`,
									)
								: '/data',
						// Carried through so double-clicking this table in turn can
						// expand it via `expandTableNode`, same as a Term's/Schema's
						// own Table neighbours elsewhere in this file.
						databaseId: table.database_id ?? undefined,
						databaseName: table.database_name ?? undefined,
						schemaId: table.schema_id ?? undefined,
						schemaName: table.schema_name ?? undefined,
					});
				});

				detailsResponse.data.sql_attributes.forEach((attribute) => {
					// Reuses the same `attribute:${id}` node id a Term's own
					// expansion would use for this SqlAttribute (see
					// `expandTermNode`), so the two share one node instead of
					// duplicating it.
					const attributeNodeId = `attribute:${attribute.id}`;
					expansionNodes.push({
						id: attributeNodeId,
						kind: 'sqlAttribute',
						label: attribute.name ?? '',
					});
					// The real direction is `(SqlAttribute)-[:HAS_SQL]->(Sql)` — drawn
					// from the Sql node regardless, same as every other structural
					// edge `addExpansion` grafts on.
					expansionEdges.push({ source: sqlEntity.id, target: attributeNodeId });
					newEntities.set(attributeNodeId, {
						id: attributeNodeId,
						kind: 'sqlAttribute',
						name: attribute.name ?? '',
						description: attribute.description,
						// Falls back to this Sql node's own `viewHref` when the
						// attribute somehow has no owning Term to link to instead —
						// same fallback rationale as `expandSqlAttributeNode`'s own
						// `sql` entity reusing its parent attribute's `viewHref`.
						viewHref:
							attribute.term_id != null
								? `/terms?focus=${encodeURIComponent(attribute.term_id)}`
								: sqlEntity.viewHref,
					});
				});
			}

			if (isExpansionStale(sqlEntity.id, activeController)) return;

			activeController.addExpansion(sqlEntity.id, expansionNodes, expansionEdges);
			mergeExpandedNodes(newEntities);
		},
		[isExpansionStale, mergeExpandedNodes],
	);

	// Reverses `expandSqlNode` — see `collapseExpansion`'s shared shape.
	const collapseSqlNode = useCallback(
		(sqlId: string) => collapseExpansion(expandedSqlIdsRef, sqlId),
		[collapseExpansion],
	);

	const handleSelectNode = useCallback(
		(nodeId: string | null) => {
			setHoveredNodePosition(null);
			// Deselects whichever connection's card is currently open and
			// drops its highlight/dimming — but, unlike before, never
			// collapses that connection's own graft (`removeExpansion`):
			// selecting a node is no different from selecting any other
			// node while some *other* node's own expansion is live, which
			// has never collapsed that expansion either. A connection now
			// only ever collapses the same way a node does — an explicit
			// toggle back on the very thing that expanded it (see
			// `handleSelectEdge`) — never as a side effect of clicking
			// something else.
			setSelectedSemanticEdgeId(null);
			setHighlightedPath(null);
			setRelationshipsNodeId(null);
			setDataDetailsType(null);
			setColumnAttributesNodeId(null);
			setSqlAttributesNodeId(null);
			setActiveNodeId(nodeId);
			let nextUrl = '/exploration';
			if (layer === ExplorationLayer.Data) {
				nextUrl = nodeId
					? `/exploration?view=data&dataId=${encodeURIComponent(nodeId)}`
					: '/exploration?view=data';
			} else if (nodeId) {
				nextUrl = `/exploration?semanticId=${encodeURIComponent(nodeId)}`;
			}
			router.replace(nextUrl, { scroll: false });
			// No `controllerRef.current?.refresh()` here: `GraphCanvas`'s own
			// `activeNodeId` prop-sync effect already refreshes once the ref
			// its node/edge reducers actually read is updated (see that
			// effect's own comment). Calling `refresh()` from here instead
			// would fire *before* `setActiveNodeId` above has committed and
			// that effect has run — repainting with the *previous* node's id
			// still in the ref — and, whenever the force simulation has
			// already settled, nothing else would ever repaint over that
			// stale frame afterwards. That's what made picking a search
			// result (or an expansion's "Focus" button) visually keep the
			// ring on whatever was selected *before*, one pick behind.
			// Bring the newly-selected node into view — most relevant when
			// it was picked from the search dropdown or a deep link rather
			// than clicked directly on the visible canvas, since it may
			// otherwise sit off-screen with no visual indication it was
			// actually selected.
			if (nodeId != null) {
				controllerRef.current?.focusNode(nodeId);
			}
		},
		[layer, router],
	);

	// Single-clicking a node selects it — see `handleSelectNode` above,
	// passed straight through to `GraphCanvas` as `onClickNode` — which is
	// what opens the node's side panel. Double-clicking a node toggles its
	// own expansion on top of that (the click starting a double click
	// already selects it via `onClickNode` before this fires), exactly one
	// hop further out each time (collapsing back down the next
	// double-click) — never reaching past a node's own direct neighbours
	// into their neighbours' own neighbours in a single double-click:
	// a Data-layer Table expands to its Schema/Columns/Terms/referencing Sql
	// queries; a Semantic-layer Term expands to its Tables/related
	// Terms/Column+SQL Attributes; a Schema (from a Table's expansion)
	// expands to every other Table in it; a ColumnAttribute (from a Term's
	// or a Column's expansion) expands to its owning Column and Term; a
	// Column (from a Table's or a ColumnAttribute's expansion) expands to
	// its owning Table, its own ColumnAttribute (if it has one), and any
	// Column on either end of a FOREIGN_KEY it's part of; a SqlAttribute
	// (from a Term's expansion) expands to its own owning Term and Sql
	// query node; and a
	// Sql node (from a SqlAttribute's expansion) expands to the Columns and
	// Tables its query references plus any CustomAnalysis nodes sharing
	// that same query — the same chain the graph itself models
	// (Table-CONTAINS->Column-HAS_ATTRIBUTE->ColumnAttribute-PROPERTY_OF->Term,
	// SqlAttribute-PROPERTY_OF->Term, SqlAttribute-HAS_SQL->Sql,
	// CustomAnalysis-HAS_SQL->Sql, Sql-SQL->Table). A Column's own ColumnAttribute (a
	// second hop from the *Table* that owns the Column) is deliberately
	// left out of a Table's own expansion for this reason — double-click
	// the specific Column instead. Only a `customAnalysis` node (nothing
	// backs it but the Sql node itself, already on the graph, one hop
	// further out) is never expandable this way.
	const handleDoubleClickNode = useCallback(
		(nodeId: string) => {
			if (layer === ExplorationLayer.Semantic) {
				const termNode = semanticGraph.nodes.find(
					(node): node is ExplorationTermNode =>
						node.id === nodeId && node.layer === ExplorationLayer.Semantic,
				);
				if (termNode != null) {
					if (expandedTermIdsRef.current.has(termNode.id)) {
						collapseTermNode(termNode.id);
					} else {
						void expandTermNode({ id: termNode.id });
					}
					return;
				}
			} else {
				const tableNode = dataGraph.nodes.find(
					(node): node is ExplorationDataNode =>
						node.id === nodeId && node.layer === ExplorationLayer.Data,
				);
				if (tableNode != null) {
					if (expandedTableIdsRef.current.has(tableNode.id)) {
						collapseTableNode(tableNode.id);
					} else {
						void expandTableNode(tableNode);
					}
					return;
				}
			}
			// Falls through here for any node not on the current layer's own
			// base graph — a Schema/Column/Term/Table/ColumnAttribute/
			// SqlAttribute/Sql/CustomAnalysis previously grafted on by some
			// node's own expansion, on either layer (e.g. a Term's Table
			// neighbours grafted on the Semantic layer — see `expandTermNode`).
			const expansionEntity = expandedNodesByIdRef.current.get(nodeId);
			if (expansionEntity == null) return;
			switch (expansionEntity.kind) {
				case 'schema':
					if (expandedSchemaIdsRef.current.has(expansionEntity.id)) {
						collapseSchemaNode(expansionEntity.id);
					} else {
						void expandSchemaNode(expansionEntity);
					}
					return;
				case 'table':
					if (expandedTableIdsRef.current.has(expansionEntity.id)) {
						collapseTableNode(expansionEntity.id);
					} else if (
						expansionEntity.databaseId != null &&
						expansionEntity.schemaId != null
					) {
						// Only a Term's/Schema's/Column's Table neighbours reach
						// this case (see `expandTermNode`/`expandSchemaNode`/
						// `expandColumnNode`) — a Data-layer base graph table is
						// always caught by the `tableNode` lookup above instead.
						// Only the raw ids are actually required to graft this
						// table's own Schema node on (`expandTableNode` only
						// reads the *name* fields for that node's caption, never
						// to fetch anything) — falling back to the id itself
						// keeps every such table expandable even when its
						// origin (e.g. a ColumnAttribute's bare `db_id`/
						// `schema_id`, with no matching name on record) never
						// carried a display name for it. Its ids can still
						// genuinely be missing (a table with no recorded
						// database/schema at all), in which case it simply
						// stays unexpandable, like a `columnAttribute`/`column`
						// entity missing its own owner below.
						void expandTableNode({
							id: expansionEntity.id,
							name: expansionEntity.name,
							databaseId: expansionEntity.databaseId,
							databaseName:
								expansionEntity.databaseName ?? expansionEntity.databaseId,
							schemaId: expansionEntity.schemaId,
							schemaName: expansionEntity.schemaName ?? expansionEntity.schemaId,
						});
					}
					return;
				case 'term':
					if (expandedTermIdsRef.current.has(expansionEntity.id)) {
						collapseTermNode(expansionEntity.id);
					} else {
						void expandTermNode(expansionEntity);
					}
					return;
				case 'columnAttribute':
					if (expandedColumnAttributeIdsRef.current.has(expansionEntity.id)) {
						collapseColumnAttributeNode(expansionEntity.id);
					} else {
						void expandColumnAttributeNode(expansionEntity);
					}
					return;
				case 'column':
					if (expandedColumnIdsRef.current.has(expansionEntity.id)) {
						collapseColumnNode(expansionEntity.id);
					} else {
						void expandColumnNode(expansionEntity);
					}
					return;
				case 'sqlAttribute':
					if (expandedSqlAttributeIdsRef.current.has(expansionEntity.id)) {
						collapseSqlAttributeNode(expansionEntity.id);
					} else {
						void expandSqlAttributeNode(expansionEntity);
					}
					return;
				case 'sql':
					if (expandedSqlIdsRef.current.has(expansionEntity.id)) {
						collapseSqlNode(expansionEntity.id);
					} else {
						void expandSqlNode(expansionEntity);
					}
					return;
				default:
					return;
			}
		},
		[
			layer,
			dataGraph.nodes,
			semanticGraph.nodes,
			collapseTableNode,
			expandTableNode,
			collapseSchemaNode,
			expandSchemaNode,
			collapseTermNode,
			expandTermNode,
			collapseColumnAttributeNode,
			expandColumnAttributeNode,
			collapseColumnNode,
			expandColumnNode,
			collapseSqlAttributeNode,
			expandSqlAttributeNode,
			collapseSqlNode,
			expandSqlNode,
		],
	);

	// Grafts the real hop chain behind a clicked semantic edge onto the live
	// graph — see `linkPathOriginId`/`pathNodeGraphId` above. The mirror
	// image of `expandTermNode`/etc: `expandedConnectionIdsRef` guards
	// against re-fetching/re-adding a second click of an already-expanded
	// connection, and `pendingCollapseIdsRef` (shared with every other
	// `expandXNode` above — ids never collide across them, and this one's
	// `linkPathOriginId` prefix guarantees that) covers the same
	// collapsed-while-in-flight race. Unlike the single shared
	// `LINK_PATH_ORIGIN_ID` this used to graft under, every connection now
	// gets its own origin id, so expanding one never disturbs another's
	// already-live graft — the same "expanding something doesn't collapse
	// anything else" guarantee every other `expandXNode` above already
	// gives node expansions.
	const expandSemanticConnection = useCallback(
		async (sourceTermId: string, targetTermId: string, edgeId: string) => {
			const activeController = controllerRef.current;
			if (activeController == null || expandedConnectionIdsRef.current.has(edgeId)) return;
			expandedConnectionIdsRef.current.add(edgeId);

			const originId = linkPathOriginId(edgeId);
			const response = await explorationApi.getSemanticLinkPath(sourceTermId, targetTermId);
			const hops = response.error ? [] : (response.data?.hops ?? []);

			const expansionNodes: ExpansionNodeInput[] = [];
			const expansionEdges: ExpansionEdgeInput[] = [];
			const seenNodeIds = new Set<string>([sourceTermId, targetTermId]);
			const pathNodeIds = new Set<string>([sourceTermId, targetTermId]);
			const pathEdgeKeys = new Set<string>([edgeId]);
			// Every Table/Column/ColumnAttribute node this path grafts on,
			// keyed by its graph id — without these, double-clicking one of
			// them does nothing (`handleDoubleClickNode` falls through to
			// `expandedNodesByIdRef`, which otherwise never hears about a
			// node this graft (rather than `expandTableNode`/etc.) put on the
			// graph) and single-clicking it shows no side panel either (see
			// `activeExpansionEntity`).
			const newEntities = new Map<string, ExpansionEntity>();

			hops.forEach((hop) => {
				[hop.source, hop.target].forEach((side) => {
					const graphId = pathNodeGraphId(side);
					pathNodeIds.add(graphId);
					if (side.type === 'term' || seenNodeIds.has(graphId)) return;
					seenNodeIds.add(graphId);
					if (!isPathNodeType(side.type)) return;
					expansionNodes.push({
						id: graphId,
						kind: side.type,
						label: side.name ?? side.id,
					});
					// A `table`/`column` side carries its own database/schema
					// ids (a `column` its owning `table_id`/`table_name` too —
					// see `_enrich_catalog_path_nodes` in
					// `auto_ontology/dal/attributes.py`) so either can be expanded
					// onward exactly like a Term's/Schema's/Column's own
					// Table (or a Table's own Column) neighbours elsewhere in
					// this file — `columnAttribute` never does (no such
					// lookup exists for it here), so it stays viewable but
					// not further expandable, same as elsewhere in this file
					// when an entity's own owner ids are missing.
					let viewHref = '/data';
					if (
						side.type === 'table' &&
						side.database_id != null &&
						side.schema_id != null
					) {
						viewHref = catalogPathFromFocusId(
							`${side.database_id}|${side.schema_id}|${side.id}`,
						);
					} else if (
						side.type === 'column' &&
						side.database_id != null &&
						side.schema_id != null &&
						side.table_id != null
					) {
						viewHref = catalogPathFromFocusId(
							`${side.database_id}|${side.schema_id}|${side.table_id}|${side.id}`,
						);
					}
					newEntities.set(graphId, {
						id: graphId,
						kind: side.type,
						name: side.name ?? side.id,
						description: null,
						viewHref,
						databaseId: side.database_id ?? undefined,
						databaseName: side.database_name ?? undefined,
						schemaId: side.schema_id ?? undefined,
						schemaName: side.schema_name ?? undefined,
						tableId: side.type === 'column' ? (side.table_id ?? undefined) : undefined,
						tableName:
							side.type === 'column' ? (side.table_name ?? undefined) : undefined,
					});
				});
				const sourceGraphId = pathNodeGraphId(hop.source);
				const targetGraphId = pathNodeGraphId(hop.target);
				expansionEdges.push({ source: sourceGraphId, target: targetGraphId });
				// Both directions, since a hop node reused from an unrelated,
				// already-expanded origin (e.g. `expandTermNode`'s own Table)
				// may have had this exact edge added in the opposite order —
				// `GraphController.addExpansion` skips re-adding an edge that
				// already exists (see its own doc comment) but never renames
				// the key it already has, so the edge's *actual* graphology
				// key can be either.
				pathEdgeKeys.add(`${sourceGraphId}:${targetGraphId}`);
				pathEdgeKeys.add(`${targetGraphId}:${sourceGraphId}`);
			});

			if (isExpansionStale(originId, activeController)) return;

			activeController.addExpansion(originId, expansionNodes, expansionEdges);
			mergeExpandedNodes(newEntities);
			setConnectionHopsByEdgeId((previous) => {
				const next = new Map(previous);
				next.set(edgeId, hops);
				return next;
			});
			// Only touches the shared `highlightedPath` when this edge is
			// still the one actually selected — the user may have already
			// clicked away (or onto a different connection) by the time this
			// resolves, and unlike the graft above (which always completes,
			// so a connection expanded this way reliably stays expanded),
			// the *highlight* is a single shared "whichever connection's
			// card is open" spotlight that a stale response must never
			// clobber out from under a newer selection.
			//
			// No explicit `controllerRef.current?.refresh()` here: it would
			// fire before this `setHighlightedPath` commits and `GraphCanvas`'s
			// own `highlightedPath` prop-sync effect updates the ref its
			// reducers actually read — see that effect's own comment, and
			// `handleSelectNode`'s, for why that stale-ref race is a bug
			// rather than a harmless extra repaint.
			if (selectedSemanticEdgeIdRef.current === edgeId) {
				setHighlightedPath({
					nodeIds: Array.from(pathNodeIds),
					edgeKeys: Array.from(pathEdgeKeys),
				});
			}
		},
		[isExpansionStale, mergeExpandedNodes],
	);

	// Reverses `expandSemanticConnection` — see `collapseExpansion`'s
	// shared shape (here, `edgeId` is what `expandedConnectionIdsRef`
	// tracks, but `linkPathOriginId(edgeId)` is what the controller's own
	// graft is keyed by — see `expandSemanticConnection`). Also drops this
	// connection's own cached hops (the next expand re-fetches them fresh).
	const collapseSemanticConnection = useCallback(
		(edgeId: string) => {
			const resolved = collapseExpansion(
				expandedConnectionIdsRef,
				edgeId,
				linkPathOriginId(edgeId),
			);
			if (!resolved) return;
			setConnectionHopsByEdgeId((previous) => {
				if (!previous.has(edgeId)) return previous;
				const next = new Map(previous);
				next.delete(edgeId);
				return next;
			});
		},
		[collapseExpansion],
	);

	// Only ever fires for a Semantic-layer (Term↔Term) edge — a Data-layer
	// (Table↔Table) one is drawn with the click-inert `structural` kind
	// instead (see `buildGraphologyGraph` in `GraphCanvas.tsx`), since that
	// edge's own SQL query/foreign key details are no longer surfaced here.
	// Clicking a connection toggles its own expansion exactly like
	// double-clicking a node does (see `handleDoubleClickNode`): clicking an
	// already-expanded connection collapses it back down, and clicking a
	// not-yet-expanded one expands it — without collapsing any *other*
	// connection (or node) that's already expanded, same guarantee every
	// other expansion on this graph gives.
	const handleSelectEdge = useCallback(
		(edgeId: string) => {
			setActiveNodeId(null);
			if (expandedConnectionIdsRef.current.has(edgeId)) {
				collapseSemanticConnection(edgeId);
				if (selectedSemanticEdgeIdRef.current === edgeId) {
					setSelectedSemanticEdgeId(null);
					setHighlightedPath(null);
				}
				return;
			}

			setSelectedSemanticEdgeId(edgeId);
			const link = semanticGraph.links.find(
				(candidate) => `${candidate.source}:${candidate.target}` === edgeId,
			);
			if (link == null) {
				setHighlightedPath(null);
				return;
			}
			// Highlights just the two terms/edge immediately, then widens to
			// the full hop chain once `expandSemanticConnection` resolves —
			// so there's no flash of a fully-undimmed graph while that
			// request is in flight.
			setHighlightedPath({ nodeIds: [link.source, link.target], edgeKeys: [edgeId] });
			void expandSemanticConnection(link.source, link.target, edgeId);
		},
		[semanticGraph.links, expandSemanticConnection, collapseSemanticConnection],
	);

	const handleToggleLayer = useCallback(() => {
		setSearch('');
		setActiveNodeId(null);
		setHoveredNodePosition(null);
		setSelectedSemanticEdgeId(null);
		setHighlightedPath(null);
		setRelationshipsNodeId(null);
		setDataDetailsType(null);
		setColumnAttributesNodeId(null);
		setSqlAttributesNodeId(null);
		// `GraphCanvas`'s own effect tears down and rebuilds its `graphology`/
		// controller from scratch whenever `graph` changes (see its own doc
		// comment) — switching layer always does, since `graph` is
		// `dataGraph`/`semanticGraph` depending on it. These refs otherwise
		// keep marking nodes as "already expanded" from the layer just left,
		// so re-expanding the very same node on the freshly-rebuilt (graft-
		// free) canvas after switching back would silently no-op. Iterates
		// `expandedIdsRefs` (every one of them, together) rather than
		// `.clear()`-ing each by name, so a future expansion kind's ref
		// can't be added elsewhere and forgotten here.
		expandedIdsRefs.current.forEach((expandedIdsRef) => expandedIdsRef.current.clear());
		// Also drops any id flagged mid-flight (collapsed before its own
		// `expandXNode` fetch resolved — see `collapseExpansion`) on the
		// layer just left. Left uncleared, a stale entry here would silently
		// swallow a *later*, unrelated `expandXNode` call for the same id
		// after switching back to this layer and re-expanding it — its own
		// pending-cancel guard would find this leftover entry and bail
		// without grafting anything, even though nothing is actually pending
		// anymore.
		pendingCollapseIdsRef.current.clear();

		setExpandedNodesById(new Map());
		setConnectionHopsByEdgeId(new Map());
		router.replace(
			layer === ExplorationLayer.Semantic ? '/exploration?view=data' : '/exploration',
			{ scroll: false },
		);
	}, [layer, router]);

	const handleControllerChange = useCallback((nextController: GraphController | null) => {
		controllerRef.current = nextController;
		setController(nextController);
	}, []);
	const handleHoverNode = useCallback((hoveredNode: HoveredNode | null) => {
		setHoveredNodePosition(hoveredNode);
	}, []);

	const graph = layer === ExplorationLayer.Semantic ? semanticGraph : dataGraph;
	const loading = layer === ExplorationLayer.Semantic ? semanticLoading : dataLoading;
	const error = layer === ExplorationLayer.Semantic ? semanticError : dataError;

	const filteredNodes = useMemo(() => {
		const query = search.trim().toLowerCase();
		if (query === '') return [];
		return graph.nodes.filter((node) => node.name.toLowerCase().includes(query));
	}, [graph.nodes, search]);

	const activeNode = useMemo(
		() => graph.nodes.find((node) => node.id === activeNodeId) ?? null,
		[activeNodeId, graph.nodes],
	);
	// Schema/Column/Term/Table nodes grafted on by `expandTableNode`/
	// `expandTermNode` aren't part of `graph.nodes`, so they fall back to the
	// lookup map populated alongside the graph mutation itself.
	const activeExpansionEntity = useMemo(
		() =>
			activeNode == null && activeNodeId != null
				? (expandedNodesById.get(activeNodeId) ?? null)
				: null,
		[activeNode, activeNodeId, expandedNodesById],
	);
	// A Term expansion entity (see `expandTableNode`) only ever carries its
	// bare name/description — `semanticGraph` is loaded unconditionally up
	// front regardless of `layer` (see the effect above) and already has the
	// full Term record for every one of them, so the richer `ActiveTermCard`
	// (Related Terms/Attribute Columns/SQL Attributes) can be shown for it
	// exactly like it is on the Semantic layer, instead of the trimmed
	// `ActiveExpansionCard` other expansion kinds fall back to below.
	const activeExpansionTermNode = useMemo(
		() =>
			activeExpansionEntity?.kind === 'term'
				? (semanticGraph.nodes.find(
						(node): node is ExplorationTermNode => node.id === activeExpansionEntity.id,
					) ?? null)
				: null,
		[activeExpansionEntity, semanticGraph.nodes],
	);
	// A related/focused entity can be either a Table (Data layer, or a Term
	// expansion's own Table nodes) or a Term (Semantic layer, or a Table
	// expansion's Term nodes) regardless of which layer is currently active,
	// so both base graphs are searched rather than just the active `graph`.
	const relationshipsNode = useMemo(
		() =>
			dataGraph.nodes.find((node) => node.id === relationshipsNodeId) ??
			semanticGraph.nodes.find((node) => node.id === relationshipsNodeId) ??
			null,
		[dataGraph.nodes, semanticGraph.nodes, relationshipsNodeId],
	);
	// Includes expansion-grafted node ids alongside the base graph's own, so
	// "Focus" in `RelationshipsModal` also works for e.g. a Term reached from
	// an expanded Table that isn't itself one of the base graph's nodes.
	const graphNodeIds = useMemo(
		() => new Set([...graph.nodes.map((node) => node.id), ...expandedNodesById.keys()]),
		[graph.nodes, expandedNodesById],
	);
	// Attribute Columns/SQL Attributes only ever belong to a Term, so these
	// two look themselves up directly in `semanticGraph` — unlike
	// `relationshipsNode` above, there's no Data-layer counterpart to also
	// check.
	const columnAttributesNode = useMemo(
		() => semanticGraph.nodes.find((node) => node.id === columnAttributesNodeId) ?? null,
		[columnAttributesNodeId, semanticGraph.nodes],
	);
	const sqlAttributesNode = useMemo(
		() => semanticGraph.nodes.find((node) => node.id === sqlAttributesNodeId) ?? null,
		[semanticGraph.nodes, sqlAttributesNodeId],
	);
	const hoveredNode = useMemo(
		() => graph.nodes.find((node) => node.id === hoveredNodePosition?.id) ?? null,
		[graph.nodes, hoveredNodePosition?.id],
	);
	const selectedSemanticEdge = useMemo(
		() =>
			graph.links.find(
				(link) => `${link.source}:${link.target}` === selectedSemanticEdgeId,
			) ?? null,
		[graph.links, selectedSemanticEdgeId],
	);
	// Memoized (rather than plain `const`s recomputed every render) since
	// `graph.nodes` is typically hundreds of nodes and every hover move
	// re-renders this component via `handleHoverNode`'s `setHoveredNodePosition`
	// — without this, moving the mouse re-scanned it twice on every single
	// frame for a value that only actually changes when the graph itself or
	// the selected connection does.
	const semanticEdgeSourceName = useMemo(
		() => graph.nodes.find((node) => node.id === selectedSemanticEdge?.source)?.name ?? '',
		[graph.nodes, selectedSemanticEdge],
	);
	const semanticEdgeTargetName = useMemo(
		() => graph.nodes.find((node) => node.id === selectedSemanticEdge?.target)?.name ?? '',
		[graph.nodes, selectedSemanticEdge],
	);
	// The `connection`-kind entity `ActiveExpansionCard` shows for a clicked
	// Semantic-layer edge — reuses that same card/kind (see its own
	// `ExpansionEntityKind` doc comment) for one consistent card shape
	// across every Exploration side panel. A Data-layer (Table↔Table) edge
	// no longer shows anything on click at all (see `buildGraphologyGraph`
	// in `GraphCanvas.tsx`). `viewHref` is never read: the card skips its
	// own "View" button entirely for this kind.
	const activeSemanticConnectionEntity: ExpansionEntity | null = useMemo(
		() =>
			selectedSemanticEdge == null
				? null
				: {
						id: selectedSemanticEdgeId ?? '',
						kind: 'connection',
						name: `${semanticEdgeSourceName} ↔ ${semanticEdgeTargetName}`,
						description: null,
						viewHref: '',
						relationshipTypes: selectedSemanticEdge.relationshipTypes,
						connectionHops:
							selectedSemanticEdgeId != null
								? (connectionHopsByEdgeId.get(selectedSemanticEdgeId) ?? null)
								: null,
						connectionSource: {
							id: selectedSemanticEdge.source,
							name: semanticEdgeSourceName,
						},
						connectionTarget: {
							id: selectedSemanticEdge.target,
							name: semanticEdgeTargetName,
						},
					},
		[
			selectedSemanticEdge,
			selectedSemanticEdgeId,
			semanticEdgeSourceName,
			semanticEdgeTargetName,
			connectionHopsByEdgeId,
		],
	);

	// Counts every Data-layer node's `nodeType` in one pass over `graph.nodes`
	// rather than the bottom-right legend running its own separate
	// `graph.nodes.filter(...).length` per `TableType` (three full scans) —
	// same "recomputed on every hover" waste `semanticEdgeSourceName`/
	// `semanticEdgeTargetName` above had, since this component re-renders on
	// every `handleHoverNode` call. Only actually used by the Data-layer
	// legend below, so it skips the scan entirely on the Semantic layer.
	const dataLayerTableTypeCounts = useMemo(() => {
		if (layer !== ExplorationLayer.Data) return null;
		const counts: Record<TableType, number> = {
			[TableType.BASE_TABLE]: 0,
			[TableType.VIEW]: 0,
			[TableType.MATERIALIZED_VIEW]: 0,
		};
		graph.nodes.forEach((node) => {
			if (node.layer === ExplorationLayer.Data) counts[node.nodeType] += 1;
		});
		return counts;
	}, [graph.nodes, layer]);

	const graphBackground =
		'bg-[radial-gradient(circle,#e4e4e7_1px,transparent_1px)] bg-[size:8px_8px] dark:bg-[radial-gradient(circle,#3f3f46_1px,transparent_1px)]';

	return (
		<main
			className={`relative h-full min-h-0 w-full overflow-hidden bg-zinc-50 dark:bg-zinc-950 ${graphBackground}`}
		>
			{/* `pointer-events-none` on the row (re-enabled per child below) so the
			empty space `justify-between` leaves in the middle of this strip lets
			clicks/drags on the canvas through to Sigma instead of being swallowed
			by this absolutely-positioned row's own invisible hit box — otherwise
			any node whose top edge falls under this strip becomes undraggable. */}
			<div className="pointer-events-none absolute left-4 right-4 top-4 z-20 flex items-start justify-between gap-4">
				<div className="pointer-events-auto flex items-start gap-2">
					<div className="relative w-72">
						<SearchInput
							value={search}
							onChange={setSearch}
							placeholder={
								layer === ExplorationLayer.Semantic
									? 'Search terms…'
									: 'Search data objects…'
							}
							aria-label={
								layer === ExplorationLayer.Semantic
									? 'Search terms in exploration'
									: 'Search data objects in exploration'
							}
							className="h-10 bg-white shadow-md dark:bg-zinc-900"
						/>
						{filteredNodes.length > 0 && (
							<ul className="absolute top-12 max-h-[calc(100dvh-8.5rem)] w-full overflow-y-auto rounded-lg border border-zinc-200 bg-white py-1 shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
								{filteredNodes.map((node) => (
									<li key={node.id}>
										<SelectButton
											theme={SelectButtonTheme.ListItem}
											onClick={() => {
												setSearch('');
												handleSelectNode(node.id);
											}}
										>
											<Icon
												name={
													node.layer === ExplorationLayer.Semantic
														? IconName.Terms
														: catalogNodeInfo[node.nodeType].icon
												}
												className={`h-4 w-4 shrink-0 ${
													node.layer === ExplorationLayer.Semantic
														? 'text-[#76b900]'
														: 'text-[#3b82b6]'
												}`}
											/>
											<Text text={node.name} />
										</SelectButton>
									</li>
								))}
							</ul>
						)}
					</div>
					<ViewToggle layer={layer} onToggle={handleToggleLayer} />
				</div>
				<div className="pointer-events-auto">
					<ZoomControls controller={controller} />
				</div>
			</div>

			{loading && <ExplorationLoader overlay />}

			{!loading && error != null && (
				<div className="flex h-full items-center justify-center px-6 text-center">
					<div className="rounded-xl border border-red-200 bg-white px-8 py-6 text-sm text-red-700 shadow-lg dark:border-red-900/50 dark:bg-zinc-900 dark:text-red-300">
						Couldn&apos;t load exploration: {error}
					</div>
				</div>
			)}

			{!loading && error == null && graph.nodes.length === 0 && (
				<EmptyState
					variant={EmptyStateVariant.Borderless}
					icon={IconName.Exploration}
					title={
						layer === ExplorationLayer.Semantic
							? 'No Terms Created Yet'
							: 'No Data Objects Found'
					}
				/>
			)}

			{!loading && error == null && graph.nodes.length > 0 && (
				<GraphCanvas
					graph={graph}
					activeNodeId={activeNodeId}
					highlightedPath={highlightedPath}
					onSelectNode={handleSelectNode}
					onSelectEdge={handleSelectEdge}
					onClickNode={handleSelectNode}
					onDoubleClickNode={handleDoubleClickNode}
					onHoverNode={handleHoverNode}
					onControllerChange={handleControllerChange}
				/>
			)}

			{hoveredNode != null && hoveredNodePosition != null && (
				<HoverNodeCard
					node={hoveredNode}
					x={hoveredNodePosition.x}
					y={hoveredNodePosition.y}
				/>
			)}

			{activeNode?.layer === ExplorationLayer.Semantic && (
				<ActiveTermCard
					key={activeNode.id}
					node={activeNode}
					onClose={() => handleSelectNode(null)}
					onView={() => router.push(`/terms?focus=${encodeURIComponent(activeNode.id)}`)}
					onShowRelationships={() => setRelationshipsNodeId(activeNode.id)}
					onShowColumnAttributes={() => setColumnAttributesNodeId(activeNode.id)}
					onShowSqlAttributes={() => setSqlAttributesNodeId(activeNode.id)}
				/>
			)}

			{activeNode?.layer === ExplorationLayer.Data && (
				<ActiveDataCard
					key={activeNode.id}
					node={activeNode}
					onClose={() => handleSelectNode(null)}
					onView={() =>
						router.push(
							`/data?focus=${encodeURIComponent(
								`${activeNode.databaseId}|${activeNode.schemaId}|${activeNode.id}`,
							)}`,
						)
					}
					onShowRelationships={() => setRelationshipsNodeId(activeNode.id)}
					onShowColumns={() => setDataDetailsType('columns')}
					onShowQueries={() => setDataDetailsType('queries')}
					onShowTerms={() => setDataDetailsType('terms')}
				/>
			)}

			{activeNode == null && activeExpansionTermNode != null && (
				<ActiveTermCard
					key={activeExpansionTermNode.id}
					node={activeExpansionTermNode}
					onClose={() => handleSelectNode(null)}
					onView={() =>
						router.push(
							`/terms?focus=${encodeURIComponent(activeExpansionTermNode.id)}`,
						)
					}
					onShowRelationships={() => setRelationshipsNodeId(activeExpansionTermNode.id)}
					onShowColumnAttributes={() =>
						setColumnAttributesNodeId(activeExpansionTermNode.id)
					}
					onShowSqlAttributes={() => setSqlAttributesNodeId(activeExpansionTermNode.id)}
				/>
			)}

			{activeNode == null &&
				activeExpansionTermNode == null &&
				activeExpansionEntity != null && (
					<ActiveExpansionCard
						key={activeExpansionEntity.id}
						node={activeExpansionEntity}
						onClose={() => handleSelectNode(null)}
						onView={() => router.push(activeExpansionEntity.viewHref)}
					/>
				)}

			<div className="absolute bottom-4 right-4 z-20 rounded-lg border border-zinc-200 bg-white px-3 py-2 shadow-lg dark:border-zinc-700 dark:bg-zinc-900">
				<p className="text-xs text-secondary dark:text-zinc-400">
					Viewing:{' '}
					{layer === ExplorationLayer.Semantic ? 'Semantic Objects' : 'Data Objects'}
				</p>
				{layer === ExplorationLayer.Semantic ? (
					<div className="mt-2 flex flex-col gap-1.5">
						<LegendItem
							kind="term"
							icon={IconName.Terms}
							label="Terms"
							count={graph.nodes.length}
						/>
						<div className="flex items-center gap-3">
							<LegendItem
								kind="columnAttribute"
								icon={IconName.Key}
								label="Column Attribute"
							/>
							<LegendItem
								kind="sqlAttribute"
								icon={IconName.ChartLine}
								label="SQL Attribute"
							/>
						</div>
						{/* A Term's own expansion (see `expandTermNode`) grafts on Data
						objects too — its Tables directly, and (via a Table/
						ColumnAttribute in turn) their Schema/Columns and any Sql query
						behind a SqlAttribute (which can, in turn, graft the
						CustomAnalysis sharing that exact Sql node — see
						`expandSqlNode`) — so this legend covers those same five
						Data-layer kinds here as well, same as the Data-layer section
						below. */}
						<div className="flex items-center gap-3">
							<LegendItem kind="table" icon={IconName.Table} label="Table" />
							<LegendItem kind="schema" icon={IconName.Schema} label="Schema" />
							<LegendItem kind="column" icon={IconName.Column} label="Column" />
						</div>
						<div className="flex items-center gap-3">
							<LegendItem kind="sql" icon={IconName.CodeBracket} label="SQL" />
							<LegendItem
								kind="customAnalysis"
								icon={IconName.ChartBar}
								label="Custom Analysis"
							/>
						</div>
					</div>
				) : (
					<div className="mt-2 flex flex-col gap-1.5">
						<div className="flex items-center gap-3">
							{[
								{ type: TableType.BASE_TABLE, label: 'Tables' },
								{ type: TableType.VIEW, label: 'Views' },
								{ type: TableType.MATERIALIZED_VIEW, label: 'Materialized' },
							].map((item) => (
								<LegendItem
									key={item.type}
									kind="table"
									icon={catalogNodeInfo[item.type].icon}
									label={item.label}
									count={dataLayerTableTypeCounts?.[item.type] ?? 0}
								/>
							))}
						</div>
						<div className="flex items-center gap-3">
							<LegendItem kind="schema" icon={IconName.Schema} label="Schema" />
							<LegendItem kind="column" icon={IconName.Column} label="Column" />
							<LegendItem kind="sql" icon={IconName.CodeBracket} label="SQL" />
							<LegendItem
								kind="customAnalysis"
								icon={IconName.ChartBar}
								label="Custom Analysis"
							/>
						</div>
						{/* A Table's own expansion (see `expandTableNode`) grafts on
						Semantic objects too — its linked Terms directly, and (via a
						Term in turn) their Column/SQL Attributes — so this legend
						covers those same three Semantic-layer kinds here as well,
						same as the Semantic-layer section above. */}
						<div className="flex items-center gap-3">
							<LegendItem kind="term" icon={IconName.Terms} label="Term" />
							<LegendItem
								kind="columnAttribute"
								icon={IconName.Key}
								label="Column Attribute"
							/>
							<LegendItem
								kind="sqlAttribute"
								icon={IconName.ChartLine}
								label="SQL Attribute"
							/>
						</div>
					</div>
				)}
			</div>

			{activeSemanticConnectionEntity != null && (
				<ActiveExpansionCard
					key={selectedSemanticEdgeId}
					node={activeSemanticConnectionEntity}
					onClose={() => {
						// Closing the card only deselects/un-highlights this
						// connection — same as closing a node's own card
						// (`handleSelectNode(null)`) never collapses that
						// node's expansion. Collapsing a connection is only
						// ever the explicit toggle in `handleSelectEdge`
						// (clicking the same edge again).
						setSelectedSemanticEdgeId(null);
						setHighlightedPath(null);
					}}
					onView={() => {}}
				/>
			)}
			<RelationshipsModal
				node={relationshipsNode}
				onClose={() => setRelationshipsNodeId(null)}
				onFocus={handleSelectNode}
				focusableNodeIds={graphNodeIds}
			/>
			<DataDetailsModal
				target={activeNode?.layer === ExplorationLayer.Data ? activeNode : null}
				type={dataDetailsType}
				onClose={() => setDataDetailsType(null)}
			/>
			<ColumnAttributesModal
				term={columnAttributesNode}
				onClose={() => setColumnAttributesNodeId(null)}
			/>
			<SqlAttributesModal
				term={sqlAttributesNode}
				onClose={() => setSqlAttributesNodeId(null)}
			/>
		</main>
	);
};
