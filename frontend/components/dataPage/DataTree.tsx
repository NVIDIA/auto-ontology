// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import type { Column, Database, Schema, Table } from '@/types/datasources';
import { DataModels } from '@/enums/datasources';
import { Icon, IconName } from '@/common/icons';
import { datasources } from '@/api/datasources';
import { catalogNodeInfo } from '@/components/dataPage/catalog-node-utils';
import { catalogPathFromFocusId } from '@/lib/data/data-catalog-path';
import { splitId } from '@/lib/data/catalog-ids';
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

function columnSubtreeContainsFocus(columnFocusPath: string, focusId: string): boolean {
	return focusId === columnFocusPath;
}

function tableSubtreeContainsFocus(tableFocusPath: string, focusId: string): boolean {
	return focusId === tableFocusPath || focusId.startsWith(`${tableFocusPath}|`);
}

function schemaSubtreeContainsFocus(schemaFocusPath: string, focusId: string): boolean {
	return focusId === schemaFocusPath || focusId.startsWith(`${schemaFocusPath}|`);
}

function databaseSubtreeContainsFocus(database: Database, focusId: string): boolean {
	return focusId === database.id || focusId.startsWith(`${database.id}|`);
}

function useOpenBranch(
	containsFocus: boolean,
	selectedId: string | undefined,
	defaultOpen: boolean,
) {
	const [open, setOpen] = useState(() => defaultOpen || containsFocus);
	const [prevSelectedId, setPrevSelectedId] = useState(selectedId);
	if (selectedId !== prevSelectedId) {
		setPrevSelectedId(selectedId);
		if (containsFocus) setOpen(true);
	}
	return [open, setOpen] as const;
}

export type DataTreeProps = {
	initialDatabases: Database[];
	selectedId?: string;
	pathBase?: string;
	className?: string;
	onTreeDataUpdated?: (databases: Database[]) => void;
};

function rowClassName(selected: boolean) {
	return `flex min-h-9 cursor-pointer items-center gap-1 rounded-lg text-left text-sm no-underline transition-colors ${
		selected
			? 'bg-[#76b900]/15 font-medium text-zinc-900 shadow-sm ring-1 ring-[#76b900]/30 dark:text-zinc-100'
			: 'text-zinc-800 hover:bg-zinc-100/90 dark:text-zinc-200 dark:hover:bg-zinc-800/70'
	}`;
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
	onClick,
	onActivateBranch,
	onChevronFocusSync,
}: {
	depth: number;
	open: boolean;
	onToggle: () => void;
	hasChildren: boolean;
	loading?: boolean;
	name: string;
	icon?: IconName;
	title?: string;
	selected: boolean;
	href?: string;
	onClick?: () => void;
	/** Branch rows: focus in URL + expand; repeat click on selected row collapses children only. */
	onActivateBranch?: () => void;
	/** After chevron toggles expand/collapse, sync `focus` to this node (same as row select). */
	onChevronFocusSync?: () => void;
}) {
	const pad = 12 + depth * 12;
	const style = { paddingLeft: pad, paddingRight: 12 };

	const nodeIconEl = icon ? (
		<span className="inline-flex shrink-0" title={title} aria-hidden>
			<Icon name={icon} className="h-4 w-4 text-zinc-500 dark:text-zinc-400" />
		</span>
	) : null;

	const chevron = (
		<span
			role="button"
			tabIndex={0}
			className="inline-flex w-4 shrink-0 cursor-pointer items-center justify-center rounded text-[10px] text-zinc-400 hover:bg-zinc-200/80 dark:hover:bg-zinc-700/80"
			onClick={(e) => {
				e.stopPropagation();
				if (!hasChildren) return;
				onToggle();
				onChevronFocusSync?.();
			}}
			onKeyDown={(e) => {
				if (e.key === 'Enter' || e.key === ' ') {
					e.preventDefault();
					e.stopPropagation();
					if (!hasChildren) return;
					onToggle();
					onChevronFocusSync?.();
				}
			}}
		>
			{loading ? '…' : hasChildren ? (open ? '▼' : '▶') : '·'}
		</span>
	);

	const nameEl = <span className="min-w-0 flex-1 truncate font-medium">{name}</span>;

	const inner = (
		<>
			{chevron}
			{nodeIconEl}
			{nameEl}
		</>
	);

	if (href && !hasChildren) {
		return (
			<Link href={href} prefetch={false} className={rowClassName(selected)} style={style}>
				{inner}
			</Link>
		);
	}

	return (
		<div
			className={rowClassName(selected)}
			style={style}
			onClick={() => {
				if (onActivateBranch) {
					onActivateBranch();
				} else if (hasChildren) {
					onToggle();
				}
				onClick?.();
			}}
		>
			{inner}
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
	const containsFocus =
		selectedId != null && columnSubtreeContainsFocus(columnFocusPath, selectedId);
	const [open, setOpen] = useOpenBranch(containsFocus, selectedId, false);
	return (
		<div>
			<Row
				depth={depth}
				open={open}
				onToggle={() => setOpen((o) => !o)}
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
	const containsFocus =
		selectedId != null && tableSubtreeContainsFocus(tableFocusPath, selectedId);
	const [open, setOpen] = useOpenBranch(containsFocus, selectedId, false);
	const fetchedOnce = useRef(false);
	const syncTableFocus = useCallback(() => {
		router.replace(catalogPathFromFocusId(tableFocusPath, pathBase), { scroll: false });
	}, [router, pathBase, tableFocusPath]);
	const activateTable = useCallback(() => {
		if (selectedId === tableFocusPath) {
			setOpen((o) => !o);
			return;
		}
		setOpen(true);
		syncTableFocus();
	}, [selectedId, tableFocusPath, setOpen, syncTableFocus]);

	const hasChildren = table.columns_count > 0;

	useEffect(() => {
		if (open && !fetchedOnce.current && table.columns.length === 0 && table.columns_count > 0) {
			fetchedOnce.current = true;
			onLoadColumns(tableLoadRef);
		}
	}, [open, table.columns.length, table.columns_count, tableLoadRef, onLoadColumns]);

	const loading = open && table.columns.length === 0 && table.columns_count > 0;

	return (
		<div>
			<Row
				depth={depth}
				open={open}
				onToggle={() => setOpen((o) => !o)}
				hasChildren={hasChildren}
				loading={loading}
				name={table.name}
				icon={catalogNodeInfo[table.table_type].icon}
				title={catalogNodeInfo[table.table_type].title}
				selected={selectedId === tableFocusPath}
				href={hasChildren ? undefined : catalogPathFromFocusId(tableFocusPath, pathBase)}
				onActivateBranch={hasChildren ? activateTable : undefined}
				onChevronFocusSync={hasChildren ? syncTableFocus : undefined}
			/>
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
	const containsFocus =
		selectedId != null && schemaSubtreeContainsFocus(schemaFocusPath, selectedId);
	const [open, setOpen] = useOpenBranch(containsFocus, selectedId, false);
	const hasChildren = (schema.tables_count ?? 0) > 0;
	const [loadingTables, setLoadingTables] = useState(false);
	const syncSchemaFocus = useCallback(() => {
		router.replace(catalogPathFromFocusId(schemaFocusPath, pathBase), { scroll: false });
	}, [router, pathBase, schemaFocusPath]);
	const activateSchemaBranch = useCallback(() => {
		if (selectedId === schemaFocusPath) {
			setOpen((o) => !o);
			return;
		}
		setOpen(true);
		syncSchemaFocus();
	}, [selectedId, schemaFocusPath, setOpen, syncSchemaFocus]);

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
		};
	}, [open, schemaLoadRef, schema.tables_count, schema.tables.length, onLoadTables]);

	return (
		<div>
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
				onActivateBranch={hasChildren ? activateSchemaBranch : undefined}
				onChevronFocusSync={hasChildren ? syncSchemaFocus : undefined}
			/>
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
	database,
	selectedId,
	pathBase,
	onLoadColumns,
	onLoadSchemas,
	onLoadTables,
}: {
	database: Database;
	selectedId?: string;
	pathBase: string;
	onLoadColumns: (tableCompoundId: string) => void;
	onLoadSchemas: (dbId: string) => Promise<void>;
	onLoadTables: (schemaCompoundId: string) => Promise<void>;
}) {
	const router = useRouter();
	const containsFocus = selectedId != null && databaseSubtreeContainsFocus(database, selectedId);
	const [open, setOpen] = useOpenBranch(containsFocus, selectedId, false);
	const hasChildren = database.schemas.length > 0 || !open;
	const [loadingSchemas, setLoadingSchemas] = useState(false);
	const syncDatabaseFocus = useCallback(() => {
		router.replace(catalogPathFromFocusId(database.id, pathBase), { scroll: false });
	}, [router, pathBase, database.id]);
	const activateDatabaseBranch = useCallback(() => {
		if (selectedId === database.id) {
			setOpen((o) => !o);
			return;
		}
		setOpen(true);
		syncDatabaseFocus();
	}, [selectedId, database.id, setOpen, syncDatabaseFocus]);

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
		};
	}, [open, database.id, database.schemas.length, onLoadSchemas]);

	return (
		<div>
			<Row
				depth={0}
				open={open}
				onToggle={() => setOpen((o) => !o)}
				hasChildren={hasChildren}
				loading={loadingSchemas}
				name={database.name}
				icon={catalogNodeInfo[DataModels.DB].icon}
				title={catalogNodeInfo[DataModels.DB].title}
				selected={selectedId === database.id}
				href={undefined}
				onActivateBranch={activateDatabaseBranch}
				onChevronFocusSync={hasChildren ? syncDatabaseFocus : undefined}
			/>
			{open && hasChildren
				? database.schemas.map((s) => (
						<SchemaBlock
							key={s.id}
							depth={1}
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
		<nav
			className={`flex min-h-0 flex-1 flex-col gap-y-1 overflow-y-auto px-3 py-2 sm:px-4 ${className}`}
			aria-label="Datasource tree"
		>
			{databases.map((database) => (
				<DatabaseBlock
					key={database.id}
					database={database}
					selectedId={selectedId}
					pathBase={pathBase}
					onLoadColumns={loadTableColumns}
					onLoadSchemas={loadSchemasForDatabase}
					onLoadTables={loadTablesForSchema}
				/>
			))}
		</nav>
	);
}

export default DataTree;
