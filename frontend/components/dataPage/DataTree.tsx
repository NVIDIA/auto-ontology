// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import type { Column, Database, Schema, Table } from '@/types/datasources';
import { DataModels } from '@/enums/datasources';
import { Icon, IconName } from '@/common/icons';
import { datasources } from '@/api/datasources';
import { catalogNodeInfo } from '@/components/dataPage/catalog-node-utils';
import { getTableType } from '@/components/dataPage/get-table-type';
import { catalogPathFromFocusId } from '@/lib/data/data-catalog-path';
import { splitId } from '@/lib/data/catalog-ids';
import { Text } from '@/common/Text';
import {
	catalogStructureFingerprint,
	mergeColumnsIntoTable,
	mergeDatabaseCatalog,
	mergeSchemasIntoDatabase,
	mergeTablesIntoSchema,
} from '@/lib/data/datasource-tree-merge';

function scheduleTreeDataNotify(
	onTreeDataUpdated: ((dbs: Database[]) => void) | undefined,
	next: Database[],
): void {
	if (!onTreeDataUpdated) return;
	queueMicrotask(() => {
		onTreeDataUpdated(next);
	});
}

// True only for a *descendant* of the node. Focus on the node itself must not
// count: a branch that owns the focus has to be able to stay collapsed.
function focusIsInsideSubtree(nodeFocusPath: string, focusId: string | undefined): boolean {
	return focusId != null && focusId.startsWith(`${nodeFocusPath}|`);
}

// Remembers which nodes the user has explicitly expanded/collapsed, so a
// page refresh restores the tree as they left it instead of only ever
// re-deriving it from the `focus` URL param (which stays put on collapse).
const OPEN_STATE_STORAGE_KEY = 'auto_ontology.dataTree.openState';

function readPersistedOpenState(): Record<string, boolean> {
	if (typeof window === 'undefined') return {};
	try {
		const raw = window.localStorage.getItem(OPEN_STATE_STORAGE_KEY);
		return raw ? (JSON.parse(raw) as Record<string, boolean>) : {};
	} catch {
		return {};
	}
}

function writePersistedOpenState(nodeId: string, open: boolean): void {
	if (typeof window === 'undefined') return;
	try {
		const state = readPersistedOpenState();
		state[nodeId] = open;
		window.localStorage.setItem(OPEN_STATE_STORAGE_KEY, JSON.stringify(state));
	} catch {
		// Storage can be unavailable (private mode, quota) — expand state
		// just won't survive a refresh, which is a harmless degradation.
	}
}

// Collapsing a branch hides its whole subtree, so any previously-remembered
// "expanded" state further down would otherwise silently resurface the next
// time this branch is opened again. Force every descendant closed so
// re-expanding starts fresh, the same as the first time the page was opened.
function collapsePersistedDescendants(nodeId: string): void {
	if (typeof window === 'undefined') return;
	try {
		const state = readPersistedOpenState();
		const prefix = `${nodeId}|`;
		let changed = false;
		Object.keys(state).forEach((key) => {
			if (key.startsWith(prefix) && state[key] !== false) {
				state[key] = false;
				changed = true;
			}
		});
		if (changed) window.localStorage.setItem(OPEN_STATE_STORAGE_KEY, JSON.stringify(state));
	} catch {
		// Storage can be unavailable — same harmless degradation as above.
	}
}

function useOpenBranch(
	nodeId: string,
	focusInsideSubtree: boolean,
	selectedId: string | undefined,
	defaultOpen: boolean,
) {
	// Deliberately starts from `defaultOpen` rather than reading
	// `readPersistedOpenState()` directly here — this initializer runs
	// during SSR too, where `window` (and so any persisted state) is always
	// unavailable, so reading it here would make the very first client
	// render disagree with the server-rendered HTML and force React to
	// reconcile a hydration mismatch. Persisted state is applied once below
	// instead, using the same "adjust state during rendering" pattern
	// `prevSelectedId` already uses further down — React re-renders before
	// committing/hydrating, so the corrected value is what actually reaches
	// the DOM, never a mismatch.
	const [open, setOpenState] = useState(() => focusInsideSubtree || defaultOpen);
	const [hasAppliedPersisted, setHasAppliedPersisted] = useState(false);
	if (!hasAppliedPersisted) {
		setHasAppliedPersisted(true);
		const persisted = readPersistedOpenState()[nodeId];
		if (persisted !== undefined) setOpenState(focusInsideSubtree || persisted);
	}
	const [prevSelectedId, setPrevSelectedId] = useState(selectedId);
	if (selectedId !== prevSelectedId) {
		setPrevSelectedId(selectedId);
		// Reveal a descendant the focus moved to (deep link, detail panel). Since
		// clicking a row also moves the focus onto that row, collapsing never
		// leaves the focus below and never re-expands this branch.
		if (focusInsideSubtree) setOpenState(true);
	}
	const setOpen = useCallback(
		(value: boolean | ((prev: boolean) => boolean)) => {
			setOpenState((prev) => {
				const next = typeof value === 'function' ? value(prev) : value;
				writePersistedOpenState(nodeId, next);
				if (!next) collapsePersistedDescendants(nodeId);
				return next;
			});
		},
		[nodeId],
	);
	return [open, setOpen] as const;
}

export type DataTreeProps = {
	initialDatabases: Database[];
	selectedId?: string;
	pathBase?: string;
	className?: string;
	onTreeDataUpdated?: (databases: Database[]) => void;
};

// Each tree level steps 16px in from its parent, as the design system draws it.
const INDENT_PX = 16;

// A row is the expand arrow plus a pill holding the node's icon and name. The
// two are separate targets with their own hover: the arrow only expands and
// collapses, the pill opens the node. The pill carries the state: default,
// hover (light green) and selected (deeper green), with the icon and the name
// turning green together.
const ROW_CLASS = 'flex h-8 w-full min-w-0 items-center gap-1 py-0.5 text-sm leading-5';
const PILL_BASE =
	'flex min-w-0 cursor-pointer items-center rounded-lg text-left no-underline focus-visible:outline-2 focus-visible:outline-[#76b900]';
const PILL_DEFAULT =
	'text-[#2b3036] hover:bg-[#f1f8e5] hover:text-[#76b900] dark:text-[#f0f0f3] dark:hover:bg-[rgba(138,210,26,0.12)] dark:hover:text-[#8ad21a]';
const PILL_SELECTED =
	'bg-[#ddedbf] text-[#76b900] dark:bg-[rgba(138,210,26,0.28)] dark:text-[#8ad21a]';
const ARROW_CLASS =
	'flex size-6 shrink-0 cursor-pointer items-center justify-center rounded text-[#2b3036] hover:bg-[#f1f8e5] hover:text-[#76b900] focus-visible:outline-2 focus-visible:outline-[#76b900] dark:text-[#f0f0f3] dark:hover:bg-[rgba(138,210,26,0.12)] dark:hover:text-[#8ad21a]';

function pillClassName(selected: boolean) {
	return `${PILL_BASE} ${selected ? PILL_SELECTED : PILL_DEFAULT}`;
}

// The design library's icons, painted through a mask in the text color so they
// follow the pill's state. Spelled out in full so Tailwind finds each class.
const MASK_BASE = 'block size-6 shrink-0 bg-current [mask-position:center] [mask-repeat:no-repeat]';
const TREE_ICON_MASK: Partial<Record<IconName, string>> = {
	[IconName.Database]: '[mask-image:url(/side-panel/database.svg)]',
	[IconName.Schema]: '[mask-image:url(/side-panel/schema.svg)]',
	[IconName.Table]: '[mask-image:url(/side-panel/table.svg)]',
	[IconName.Column]: '[mask-image:url(/side-panel/column.svg)]',
};

function TreeIcon({ icon, title }: { icon: IconName; title?: string }) {
	const mask = TREE_ICON_MASK[icon];
	// Views and materialized views have no design library icon yet, so they
	// keep the app's own, centered in the same 24px slot.
	return mask ? (
		<span className={`${MASK_BASE} ${mask}`} title={title} aria-hidden />
	) : (
		<span
			className="flex size-6 shrink-0 items-center justify-center"
			title={title}
			aria-hidden
		>
			<Icon name={icon} className="size-4" />
		</span>
	);
}

/** Placeholder for a branch whose children are still being read. */
function TreeSkeletonRow({ depth }: { depth: number }) {
	return (
		<div
			role="status"
			aria-label="Loading"
			className="flex h-8 items-center gap-1 py-0.5"
			style={{ paddingLeft: depth * INDENT_PX }}
		>
			<span className="size-6 shrink-0" aria-hidden />
			<span className="flex w-44 min-w-0 animate-pulse items-center gap-1 py-1 pr-1 pl-px">
				{/* eslint-disable @next/next/no-img-element -- static design placeholder, one file per color scheme */}
				<img
					src="/side-panel/skeleton-dot-light.svg"
					alt=""
					width={22}
					height={22}
					className="block size-[22px] shrink-0 dark:hidden"
				/>
				<img
					src="/side-panel/skeleton-dot-dark.svg"
					alt=""
					width={22}
					height={22}
					className="hidden size-[22px] shrink-0 dark:block"
				/>
				{/* eslint-enable @next/next/no-img-element */}
				<span className="h-4 min-w-0 flex-1 rounded-xl bg-[#e9eaea] dark:bg-[#2e2e33]" />
			</span>
		</div>
	);
}

function Row({
	depth,
	open,
	onToggle,
	hasChildren,
	loading,
	name,
	icon,
	title,
	selected,
	href,
	onActivate,
}: {
	depth: number;
	/** Only meaningful when `hasChildren` — leaf rows can omit it. */
	open?: boolean;
	/** Expands or collapses the branch; called by the arrow only. */
	onToggle?: () => void;
	hasChildren: boolean;
	loading?: boolean;
	name: string;
	icon?: IconName;
	title?: string;
	selected: boolean;
	/** Leaf rows: where the pill links to. */
	href?: string;
	/** Branch rows: opens the node when its pill is clicked. */
	onActivate?: () => void;
}) {
	// The indent is a runtime value per depth, so it cannot be a class.
	const style = { paddingLeft: depth * INDENT_PX };

	// Leaves keep an empty arrow slot so their icons line up under a sibling
	// branch's icon, as in the design.
	const arrow = hasChildren ? (
		<button
			type="button"
			aria-expanded={open}
			aria-label={open ? `Collapse ${name}` : `Expand ${name}`}
			aria-busy={loading || undefined}
			className={ARROW_CLASS}
			onClick={() => onToggle?.()}
		>
			<span
				aria-hidden
				className={`${MASK_BASE} ${open ? '[mask-image:url(/side-panel/arrow-down.svg)]' : '[mask-image:url(/side-panel/arrow-right.svg)]'}`}
			/>
		</button>
	) : (
		<span className="size-6 shrink-0" aria-hidden />
	);

	// A pill with an icon hugs it on the left; one without pads evenly.
	const pillClass = `${pillClassName(selected)} ${icon ? 'gap-1 py-1 pr-2 pl-1' : 'px-2 py-1.5'}`;
	const pillContent = (
		<>
			{icon ? <TreeIcon icon={icon} title={title} /> : null}
			<Text text={name} />
		</>
	);

	return (
		<div className={ROW_CLASS} style={style}>
			{arrow}
			{href && !hasChildren ? (
				<Link
					href={href}
					prefetch={false}
					aria-current={selected ? 'page' : undefined}
					className={pillClass}
				>
					{pillContent}
				</Link>
			) : (
				<button
					type="button"
					aria-current={selected ? 'page' : undefined}
					className={pillClass}
					onClick={() => onActivate?.()}
				>
					{pillContent}
				</button>
			)}
		</div>
	);
}

function ColumnBlock({
	depth,
	column,
	columnFocusPath,
	selectedId,
	pathBase,
}: {
	depth: number;
	column: Column;
	columnFocusPath: string;
	selectedId?: string;
	pathBase: string;
}) {
	return (
		<div>
			<Row
				depth={depth}
				hasChildren={false}
				name={column.column_name}
				icon={catalogNodeInfo[DataModels.COLUMN].icon}
				title={catalogNodeInfo[DataModels.COLUMN].title}
				selected={selectedId === columnFocusPath}
				href={catalogPathFromFocusId(columnFocusPath, pathBase)}
			/>
		</div>
	);
}

function TableBlock({
	depth,
	table,
	tableFocusPath,
	tableLoadRef,
	selectedId,
	pathBase,
	onLoadColumns,
}: {
	depth: number;
	table: Table;
	/** `dbId|schemaId|tableId` for URLs, selection, and lazy column fetch. */
	tableFocusPath: string;
	/** Same as ``tableFocusPath``; passed to ``onLoadColumns`` for column fetch. */
	tableLoadRef: string;
	selectedId?: string;
	pathBase: string;
	onLoadColumns: (tableCompoundId: string) => void;
}) {
	const router = useRouter();
	const focusInsideTable = focusIsInsideSubtree(tableFocusPath, selectedId);
	const [open, setOpen] = useOpenBranch(tableFocusPath, focusInsideTable, selectedId, false);
	const fetchedOnce = useRef(false);
	const syncTableFocus = useCallback(() => {
		router.replace(catalogPathFromFocusId(tableFocusPath, pathBase), { scroll: false });
	}, [router, pathBase, tableFocusPath]);
	// Opening a node from its label also reveals its children; only the arrow
	// collapses a branch.
	const activateTable = useCallback(() => {
		setOpen(true);
		syncTableFocus();
	}, [setOpen, syncTableFocus]);

	const hasChildren = table.columns_count > 0;

	useEffect(() => {
		if (open && !fetchedOnce.current && table.columns.length === 0 && table.columns_count > 0) {
			fetchedOnce.current = true;
			onLoadColumns(tableLoadRef);
		}
	}, [open, table.columns.length, table.columns_count, tableLoadRef, onLoadColumns]);

	const loading = open && table.columns.length === 0 && table.columns_count > 0;
	const tableInfo = catalogNodeInfo[getTableType(table.table_type)];

	return (
		<div className="flex flex-col gap-y-1">
			<Row
				depth={depth}
				open={open}
				onToggle={() => setOpen((o) => !o)}
				hasChildren={hasChildren}
				loading={loading}
				name={table.name}
				icon={tableInfo.icon}
				title={tableInfo.title}
				selected={selectedId === tableFocusPath}
				href={hasChildren ? undefined : catalogPathFromFocusId(tableFocusPath, pathBase)}
				onActivate={activateTable}
			/>
			{loading ? <TreeSkeletonRow depth={depth + 1} /> : null}
			{open && table.columns.length > 0
				? table.columns.map((col) => (
						<ColumnBlock
							key={col.id}
							depth={depth + 1}
							column={col}
							columnFocusPath={`${tableFocusPath}|${col.id}`}
							selectedId={selectedId}
							pathBase={pathBase}
						/>
					))
				: null}
		</div>
	);
}

function SchemaBlock({
	depth,
	databaseId,
	schema,
	selectedId,
	pathBase,
	onLoadColumns,
	onLoadTables,
}: {
	depth: number;
	databaseId: string;
	schema: Schema;
	selectedId?: string;
	pathBase: string;
	onLoadColumns: (tableCompoundId: string) => void;
	onLoadTables: (schemaCompoundId: string) => Promise<void>;
}) {
	const router = useRouter();
	const schemaFocusPath = `${databaseId}|${schema.id}`;
	const schemaLoadRef = schemaFocusPath;
	const focusInsideSchema = focusIsInsideSubtree(schemaFocusPath, selectedId);
	const [open, setOpen] = useOpenBranch(schemaFocusPath, focusInsideSchema, selectedId, false);
	const hasChildren = (schema.tables_count ?? 0) > 0;
	const [loadingTables, setLoadingTables] = useState(false);
	const syncSchemaFocus = useCallback(() => {
		router.replace(catalogPathFromFocusId(schemaFocusPath, pathBase), { scroll: false });
	}, [router, pathBase, schemaFocusPath]);
	// Opening a node from its label also reveals its children; only the arrow
	// collapses a branch.
	const activateSchemaBranch = useCallback(() => {
		setOpen(true);
		syncSchemaFocus();
	}, [setOpen, syncSchemaFocus]);

	useEffect(() => {
		if (!open || schema.tables.length > 0 || (schema.tables_count ?? 0) === 0) return;
		let cancelled = false;
		void (async () => {
			await Promise.resolve();
			if (cancelled) return;
			setLoadingTables(true);
			try {
				await onLoadTables(schemaLoadRef);
			} finally {
				if (!cancelled) setLoadingTables(false);
			}
		})();
		return () => {
			cancelled = true;
			// A collapse mid-fetch skips the `finally` above, so clear the flag here.
			setLoadingTables(false);
		};
	}, [open, schemaLoadRef, schema.tables_count, schema.tables.length, onLoadTables]);

	return (
		<div className="flex flex-col gap-y-1">
			<Row
				depth={depth}
				open={open}
				onToggle={() => setOpen((o) => !o)}
				hasChildren={hasChildren}
				loading={loadingTables}
				name={schema.schema_name}
				icon={catalogNodeInfo[DataModels.SCHEMA].icon}
				title={catalogNodeInfo[DataModels.SCHEMA].title}
				selected={selectedId === schemaFocusPath}
				href={hasChildren ? undefined : catalogPathFromFocusId(schemaFocusPath, pathBase)}
				onActivate={activateSchemaBranch}
			/>
			{open && loadingTables ? <TreeSkeletonRow depth={depth + 1} /> : null}
			{open && hasChildren
				? schema.tables.map((t) => {
						const tableFocusPath = `${schemaFocusPath}|${t.id}`;
						return (
							<TableBlock
								key={t.id}
								depth={depth + 1}
								table={t}
								tableFocusPath={tableFocusPath}
								tableLoadRef={tableFocusPath}
								selectedId={selectedId}
								pathBase={pathBase}
								onLoadColumns={onLoadColumns}
							/>
						);
					})
				: null}
		</div>
	);
}

function DatabaseBlock({
	depth,
	database,
	selectedId,
	pathBase,
	onLoadColumns,
	onLoadSchemas,
	onLoadTables,
}: {
	depth: number;
	database: Database;
	selectedId?: string;
	pathBase: string;
	onLoadColumns: (tableCompoundId: string) => void;
	onLoadSchemas: (dbId: string) => Promise<void>;
	onLoadTables: (schemaCompoundId: string) => Promise<void>;
}) {
	const router = useRouter();
	const focusInsideDatabase = focusIsInsideSubtree(database.id, selectedId);
	const [open, setOpen] = useOpenBranch(database.id, focusInsideDatabase, selectedId, false);
	const hasChildren = database.schemas.length > 0 || !open;
	const [loadingSchemas, setLoadingSchemas] = useState(false);
	const syncDatabaseFocus = useCallback(() => {
		router.replace(catalogPathFromFocusId(database.id, pathBase), { scroll: false });
	}, [router, pathBase, database.id]);
	// Opening a node from its label also reveals its children; only the arrow
	// collapses a branch.
	const activateDatabaseBranch = useCallback(() => {
		setOpen(true);
		syncDatabaseFocus();
	}, [setOpen, syncDatabaseFocus]);

	useEffect(() => {
		if (!open || database.schemas.length > 0) return;
		let cancelled = false;
		void (async () => {
			await Promise.resolve();
			if (cancelled) return;
			setLoadingSchemas(true);
			try {
				await onLoadSchemas(database.id);
			} finally {
				if (!cancelled) setLoadingSchemas(false);
			}
		})();
		return () => {
			cancelled = true;
			setLoadingSchemas(false);
		};
	}, [open, database.id, database.schemas.length, onLoadSchemas]);

	return (
		<div className="flex flex-col gap-y-1">
			<Row
				depth={depth}
				open={open}
				onToggle={() => setOpen((o) => !o)}
				hasChildren={hasChildren}
				loading={loadingSchemas}
				name={database.name}
				icon={catalogNodeInfo[DataModels.DB].icon}
				title={catalogNodeInfo[DataModels.DB].title}
				selected={selectedId === database.id}
				onActivate={activateDatabaseBranch}
			/>
			{open && loadingSchemas ? <TreeSkeletonRow depth={depth + 1} /> : null}
			{open && hasChildren
				? database.schemas.map((s) => (
						<SchemaBlock
							key={s.id}
							depth={depth + 1}
							databaseId={database.id}
							schema={s}
							selectedId={selectedId}
							pathBase={pathBase}
							onLoadColumns={onLoadColumns}
							onLoadTables={onLoadTables}
						/>
					))
				: null}
		</div>
	);
}

// Open-state key for the root; no catalog id can collide with it.
const ALL_DATA_NODE_ID = '__all_data__';

/**
 * The tree's root: the page that lists every data connector, so each database
 * hangs one level under it. Its label opens that page; its arrow folds the
 * whole catalog away.
 */
function AllDataBlock({
	selectedId,
	pathBase,
	hasDatabases,
	children,
}: {
	selectedId?: string;
	pathBase: string;
	hasDatabases: boolean;
	children: ReactNode;
}) {
	const router = useRouter();
	// Every catalog node is below the root, so any focus keeps it open.
	const [open, setOpen] = useOpenBranch(ALL_DATA_NODE_ID, selectedId != null, selectedId, true);
	const activate = useCallback(() => {
		setOpen(true);
		router.replace(pathBase, { scroll: false });
	}, [router, pathBase, setOpen]);

	return (
		<div className="flex flex-col gap-y-1">
			<Row
				depth={0}
				open={open}
				onToggle={() => setOpen((o) => !o)}
				hasChildren={hasDatabases}
				name="All Data"
				selected={selectedId == null}
				onActivate={activate}
			/>
			{open ? children : null}
		</div>
	);
}

export function DataTree({
	initialDatabases,
	selectedId,
	pathBase = '/data',
	className = '',
	onTreeDataUpdated,
}: DataTreeProps) {
	const [databases, setDatabases] = useState<Database[]>(initialDatabases);
	const catalogFpRef = useRef('');

	useEffect(() => {
		const fp = catalogStructureFingerprint(initialDatabases);
		if (fp === catalogFpRef.current) return;
		catalogFpRef.current = fp;
		queueMicrotask(() => {
			setDatabases((prev) => mergeDatabaseCatalog(prev, initialDatabases));
		});
	}, [initialDatabases]);

	const loadSchemasForDatabase = useCallback(
		async (dbId: string) => {
			const res = await datasources.getSchemasForDatabase(dbId);
			if (res.error || !res.data) return;
			setDatabases((prev) => {
				const next = mergeSchemasIntoDatabase(prev, dbId, res.data);
				scheduleTreeDataNotify(onTreeDataUpdated, next);
				return next;
			});
		},
		[onTreeDataUpdated],
	);

	const loadTablesForSchema = useCallback(
		async (schemaCompoundId: string) => {
			const [, schemaElemId] = splitId(schemaCompoundId, 2);
			const res = await datasources.getTablesForSchema(schemaElemId);
			if (res.error || !res.data) return;
			setDatabases((prev) => {
				const next = mergeTablesIntoSchema(prev, schemaElemId, res.data);
				scheduleTreeDataNotify(onTreeDataUpdated, next);
				return next;
			});
		},
		[onTreeDataUpdated],
	);

	const loadTableColumns = useCallback(
		async (tableCompoundId: string) => {
			const [, , tableElemId] = splitId(tableCompoundId, 3);
			const res = await datasources.getColumnsForTable(tableElemId);
			if (res.error || !res.data) return;
			setDatabases((prev) => {
				const next = mergeColumnsIntoTable(prev, tableElemId, res.data);
				scheduleTreeDataNotify(onTreeDataUpdated, next);
				return next;
			});
		},
		[onTreeDataUpdated],
	);

	return (
		// The 16px under the panel header is padding inside the scroll area, so
		// rows scroll up to the header instead of being cut off above it.
		<nav
			className={`flex min-h-0 flex-1 flex-col gap-y-1 overflow-y-auto px-5 pt-4 pb-4 ${className}`}
			aria-label="Datasource tree"
		>
			<AllDataBlock
				selectedId={selectedId}
				pathBase={pathBase}
				hasDatabases={databases.length > 0}
			>
				{databases.map((database) => (
					<DatabaseBlock
						key={database.id}
						depth={1}
						database={database}
						selectedId={selectedId}
						pathBase={pathBase}
						onLoadColumns={loadTableColumns}
						onLoadSchemas={loadSchemasForDatabase}
						onLoadTables={loadTablesForSchema}
					/>
				))}
			</AllDataBlock>
		</nav>
	);
}

export default DataTree;
