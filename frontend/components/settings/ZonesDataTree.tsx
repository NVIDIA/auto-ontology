// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef, useState } from 'react';
import type { Database, Schema, Table } from '@/types/datasources';
import { Text } from '@/common/Text';

const treeRowClassName =
	'flex min-h-9 items-center gap-2 rounded-lg px-2 text-sm text-zinc-800 hover:bg-zinc-100/90 dark:text-zinc-200 dark:hover:bg-zinc-800/70';

export type ZonesDataTreeProps = {
	databases: Database[];
	selectedItems: Set<string>;
	onSelectedItemsChange: (next: Set<string>) => void;
	onLoadSchemas: (dbId: string) => Promise<Schema[]>;
	onLoadTables: (schemaId: string) => Promise<void>;
	expandSelectedOnlyKey?: number;
};

type SelectionState = {
	checked: boolean;
	indeterminate: boolean;
};

export const ZonesDataTree = ({
	databases,
	selectedItems,
	onSelectedItemsChange,
	onLoadSchemas,
	onLoadTables,
	expandSelectedOnlyKey,
}: ZonesDataTreeProps) => {
	const [openIds, setOpenIds] = useState<Record<string, boolean>>({});
	const [loadingIds, setLoadingIds] = useState<Record<string, boolean>>({});
	const lastAppliedExpandKeyRef = useRef<number | undefined>(undefined);

	const setLoading = (id: string, loading: boolean) => {
		setLoadingIds((prev) => ({ ...prev, [id]: loading }));
	};

	const getSchemaDescendantIds = (schema: Schema): string[] =>
		schema.tables.map((table) => table.id);

	const getDatabaseDescendantIds = (db: Database): string[] =>
		db.schemas.flatMap((schema) => [schema.id, ...getSchemaDescendantIds(schema)]);

	const getSelectionState = (id: string, descendantIds: string[]): SelectionState => {
		if (descendantIds.length === 0) {
			return { checked: selectedItems.has(id), indeterminate: false };
		}

		const selectedDescendantsCount = descendantIds.reduce(
			(count, nodeId) => count + (selectedItems.has(nodeId) ? 1 : 0),
			0,
		);
		const allDescendantsSelected = selectedDescendantsCount === descendantIds.length;
		const anyDescendantSelected = selectedDescendantsCount > 0;
		const selfSelected = selectedItems.has(id);

		return {
			checked: allDescendantsSelected || (selfSelected && !anyDescendantSelected),
			indeterminate: !allDescendantsSelected && (anyDescendantSelected || selfSelected),
		};
	};

	const toggleSelection = (
		id: string,
		descendantIds: string[],
		selectionState: SelectionState,
	) => {
		const subtreeIds = [id, ...descendantIds];
		const shouldSelectAll = !selectionState.checked;
		const next = new Set(selectedItems);
		subtreeIds.forEach((nodeId) => {
			if (shouldSelectAll) {
				next.add(nodeId);
			} else {
				next.delete(nodeId);
			}
		});
		onSelectedItemsChange(next);
	};

	const previousChildrenRef = useRef<Record<string, Set<string>>>({});

	useEffect(() => {
		const next = new Set(selectedItems);
		let hasChanges = false;
		const nextPreviousChildren: Record<string, Set<string>> = {};

		databases.forEach((db) => {
			const dbChildren = db.schemas.map((schema) => schema.id);
			const dbChildrenSet = new Set(dbChildren);
			const previousDbChildren = previousChildrenRef.current[db.id] ?? new Set<string>();
			if (next.has(db.id)) {
				dbChildren.forEach((schemaId) => {
					if (!previousDbChildren.has(schemaId) && !next.has(schemaId)) {
						next.add(schemaId);
						hasChanges = true;
					}
				});
			}
			nextPreviousChildren[db.id] = dbChildrenSet;

			db.schemas.forEach((schema) => {
				const schemaChildren = schema.tables.map((table) => table.id);
				const schemaChildrenSet = new Set(schemaChildren);
				const previousSchemaChildren =
					previousChildrenRef.current[schema.id] ?? new Set<string>();
				if (next.has(schema.id)) {
					schemaChildren.forEach((tableId) => {
						if (!previousSchemaChildren.has(tableId) && !next.has(tableId)) {
							next.add(tableId);
							hasChanges = true;
						}
					});
				}
				nextPreviousChildren[schema.id] = schemaChildrenSet;
			});
		});

		previousChildrenRef.current = nextPreviousChildren;

		if (hasChanges) {
			onSelectedItemsChange(next);
		}
	}, [databases, selectedItems, onSelectedItemsChange]);

	useEffect(() => {
		if (expandSelectedOnlyKey == null) return;
		if (lastAppliedExpandKeyRef.current === expandSelectedOnlyKey) return;
		lastAppliedExpandKeyRef.current = expandSelectedOnlyKey;

		const nextOpenIds: Record<string, boolean> = {};
		databases.forEach((db) => {
			const dbHasSelectedSchema = db.schemas.some(
				(schema) =>
					selectedItems.has(schema.id) ||
					schema.tables.some((table) => selectedItems.has(table.id)),
			);
			if (selectedItems.has(db.id) || dbHasSelectedSchema) {
				nextOpenIds[db.id] = true;
			}

			db.schemas.forEach((schema) => {
				const schemaHasSelectedTable = schema.tables.some((table) =>
					selectedItems.has(table.id),
				);
				if (selectedItems.has(schema.id) || schemaHasSelectedTable) {
					nextOpenIds[schema.id] = true;
				}
			});
		});

		setOpenIds(nextOpenIds);
	}, [databases, selectedItems, expandSelectedOnlyKey]);

	const toggleDatabase = async (db: Database) => {
		const isOpen = openIds[db.id] === true;
		const nextOpen = !isOpen;
		setOpenIds((prev) => ({ ...prev, [db.id]: nextOpen }));
		if (!nextOpen || db.schemas.length > 0) return;
		setLoading(db.id, true);
		try {
			await onLoadSchemas(db.id);
		} finally {
			setLoading(db.id, false);
		}
	};

	const toggleSchema = async (schema: Schema) => {
		const isOpen = openIds[schema.id] === true;
		const nextOpen = !isOpen;
		setOpenIds((prev) => ({ ...prev, [schema.id]: nextOpen }));
		if (!nextOpen || schema.tables.length > 0 || schema.tables_count === 0) return;
		setLoading(schema.id, true);
		try {
			await onLoadTables(schema.id);
		} finally {
			setLoading(schema.id, false);
		}
	};

	const toggleTable = (table: Table) => {
		setOpenIds((prev) => ({ ...prev, [table.id]: !(prev[table.id] === true) }));
	};

	const loadTablesForSchema = async (
		schema: Schema,
		loadingId: string = schema.id,
	): Promise<void> => {
		if (schema.tables.length > 0 || schema.tables_count === 0) return;
		setLoading(loadingId, true);
		try {
			await onLoadTables(schema.id);
		} finally {
			setLoading(loadingId, false);
		}
	};

	const getCompactDatabaseLabel = (db: Database): string => {
		const shouldCompactSchema = db.num_of_schemas === 1 && db.schemas.length === 1;
		if (!shouldCompactSchema) return db.name;
		const [schema] = db.schemas;
		if (schema.tables_count === 1 && schema.tables.length === 1) {
			return `${db.name}/${schema.schema_name}/${schema.tables[0].name}`;
		}
		return `${db.name}/${schema.schema_name}`;
	};

	const hasCompactDatabaseChildren = (db: Database): boolean => {
		if (db.num_of_schemas !== 1) return true;
		const [schema] = db.schemas;
		if (schema == null) return true;
		return schema.tables_count > 1;
	};

	const toggleCompactDatabase = async (db: Database): Promise<void> => {
		const isOpen = openIds[db.id] === true;
		const nextOpen = !isOpen;
		setOpenIds((prev) => ({ ...prev, [db.id]: nextOpen }));
		if (!nextOpen) return;
		let schemas = db.schemas;
		if (schemas.length === 0) {
			setLoading(db.id, true);
			try {
				schemas = await onLoadSchemas(db.id);
			} finally {
				setLoading(db.id, false);
			}
		}
		const [schema] = schemas;
		if (schema == null) return;
		if (schema.tables_count > 0) {
			await loadTablesForSchema(schema, db.id);
		}
	};

	const row = (
		depth: number,
		id: string,
		label: string,
		hasChildren: boolean,
		isOpen: boolean,
		selectionState: SelectionState,
		onSelectionChange: () => void,
		onToggle: () => void | Promise<void>,
	) => (
		<div key={id} className={treeRowClassName} style={{ paddingLeft: 8 + depth * 14 }}>
			<button
				type="button"
				onClick={() => {
					void onToggle();
				}}
				className="inline-flex w-4 shrink-0 cursor-pointer items-center justify-center rounded text-[10px] text-zinc-400 hover:bg-zinc-200/80 dark:hover:bg-zinc-700/80"
				aria-label={isOpen ? 'Collapse' : 'Expand'}
			>
				{loadingIds[id] ? '…' : hasChildren ? (isOpen ? '▼' : '▶') : '·'}
			</button>
			<input
				type="checkbox"
				checked={selectionState.checked}
				ref={(element) => {
					if (element) {
						element.indeterminate = selectionState.indeterminate;
					}
				}}
				onChange={onSelectionChange}
				className="h-4 w-4 cursor-pointer rounded border-zinc-300 text-[#76b900] focus:ring-[#76b900]/40 dark:border-zinc-600"
			/>
			<Text text={label} />
		</div>
	);

	const renderTableRow = (table: Table, depth: number) => {
		const tableSelectionState = getSelectionState(table.id, []);
		return (
			<div key={table.id}>
				{row(
					depth,
					table.id,
					table.name,
					false,
					false,
					tableSelectionState,
					() => toggleSelection(table.id, [], tableSelectionState),
					() => toggleTable(table),
				)}
			</div>
		);
	};

	return (
		<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-2 dark:border-zinc-700/90 dark:bg-zinc-950/40">
			<div className="max-h-[min(50vh,420px)] overflow-y-auto">
				{databases.map((db) => {
					const isCompactDb = db.num_of_schemas === 1;
					const dbOpen = openIds[db.id] === true;
					const dbDescendantIds = getDatabaseDescendantIds(db);
					const dbSelectionDescendantIds = isCompactDb
						? db.schemas.flatMap((schema) => getSchemaDescendantIds(schema))
						: dbDescendantIds;
					const dbSelectionState = getSelectionState(db.id, dbSelectionDescendantIds);
					return (
						<div key={db.id}>
							{row(
								0,
								db.id,
								isCompactDb ? getCompactDatabaseLabel(db) : db.name,
								isCompactDb ? hasCompactDatabaseChildren(db) : true,
								dbOpen,
								dbSelectionState,
								() => toggleSelection(db.id, dbDescendantIds, dbSelectionState),
								() =>
									isCompactDb ? toggleCompactDatabase(db) : toggleDatabase(db),
							)}
							{dbOpen
								? isCompactDb
									? (() => {
											const [schema] = db.schemas;
											if (schema == null) return null;
											if (schema.tables_count > 1) {
												return schema.tables.map((table) =>
													renderTableRow(table, 1),
												);
											}
											return null;
										})()
									: db.schemas.map((schema) => {
											const schemaOpen = openIds[schema.id] === true;
											const schemaDescendantIds =
												getSchemaDescendantIds(schema);
											const schemaSelectionState = getSelectionState(
												schema.id,
												schemaDescendantIds,
											);
											return (
												<div key={schema.id}>
													{row(
														1,
														schema.id,
														schema.schema_name,
														schema.tables_count > 0,
														schemaOpen,
														schemaSelectionState,
														() =>
															toggleSelection(
																schema.id,
																schemaDescendantIds,
																schemaSelectionState,
															),
														() => toggleSchema(schema),
													)}
													{schemaOpen
														? schema.tables.map((table) =>
																renderTableRow(table, 2),
															)
														: null}
												</div>
											);
										})
								: null}
						</div>
					);
				})}
			</div>
		</div>
	);
};
