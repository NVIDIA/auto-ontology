// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

import { datasources } from '@/api/datasources';
import { Button, SelectButton } from '@/common/Button';
import { DataTreeSelect } from '@/common/DataTreeSelect';
import { Icon, IconName } from '@/common/icons';
import { Popover } from '@/common/Popover';
import { ButtonTheme, SelectButtonTheme, Size } from '@/enums/button';
import { PopoverAlign } from '@/enums/popover';
import { mergeSchemasIntoDatabase } from '@/lib/data/datasource-tree-merge';
import type { Database } from '@/types/datasources';

export type DataFilterProps = {
	/** Database and schema ids in one list; null is the whole catalog. */
	selected: string[] | null;
	onChange: (selected: string[] | null) => void;
};

/** Every id the tree can currently show as ticked. */
const everyKnownId = (databases: Database[]): string[] =>
	databases.flatMap((database) => [database.id, ...database.schemas.map((schema) => schema.id)]);

/**
 * The tree's selection reduced to the ids the request should carry.
 *
 * Two things happen here, and neither can happen in the tree. It keeps a set
 * holding every ticked node, parents included, because that is what draws the
 * checkboxes — but sent as-is that set is wrong twice over.
 *
 * It is wrong because a partly ticked database keeps its own id in the set,
 * and the backend reads a database id as "everything under it": unticking one
 * schema would narrow nothing. So a database is only sent when all of it is
 * ticked, and otherwise gives way to the schemas that are.
 *
 * And it is wrong because the ids are redundant. A whole database is sent as
 * the database rather than as its schemas, which is not just shorter: it goes
 * on meaning "this database" as schemas are added to it, where a list frozen
 * from today's would quietly exclude tomorrow's.
 */
const toDataIds = (databases: Database[], selection: Set<string>): string[] => {
	const ids: string[] = [];
	databases.forEach((database) => {
		const { schemas } = database;
		const picked = schemas.filter((schema) => selection.has(schema.id));

		if (schemas.length > 0 && picked.length === schemas.length) {
			ids.push(database.id);
			return;
		}
		// Unopened, so its schemas are unknown and cannot have been unticked
		// one by one — the database's own box is the whole answer.
		if (schemas.length === 0) {
			if (selection.has(database.id)) ids.push(database.id);
			return;
		}
		picked.forEach((schema) => ids.push(schema.id));
	});
	return ids;
};

/**
 * Which part of the catalog a search may reach.
 *
 * The same tree the zones screen selects with, stopped at the schema: tables
 * and columns are what the search *returns*, so offering them here would be
 * asking somebody to find the thing they are looking for. The reference
 * implementation draws its line in the same place.
 *
 * Selection follows the other two pickers — null is everything and opens with
 * every box ticked, and a selection covering the whole catalog collapses back
 * to null so that "untouched" has one representation for the reset button to
 * compare against.
 *
 * Unlike Objects and Tags, this filter also narrows what *kinds* come back:
 * only tables, views and columns live under a schema. The backend does that
 * (`_data_filter_rules_out`) and the panel does not restate it — the Objects
 * picker goes on offering every kind, and the tabs report what was found.
 */
export const DataFilter = ({ selected, onChange }: DataFilterProps) => {
	const [databases, setDatabases] = useState<Database[]>([]);
	const [loading, setLoading] = useState(true);
	const [failed, setFailed] = useState(false);

	useEffect(() => {
		let cancelled = false;
		void datasources.getDBs().then((response) => {
			if (cancelled) return;
			setFailed(response.error === true);
			setDatabases(response.data ?? []);
			setLoading(false);
		});
		return () => {
			cancelled = true;
		};
	}, []);

	const loadSchemas = useCallback(async (dbId: string) => {
		const response = await datasources.getSchemasForDatabase(dbId);
		if (response.error || !response.data) return [];
		setDatabases((prev) => mergeSchemasIntoDatabase(prev, dbId, response.data ?? []));
		return response.data;
	}, []);

	// Derived rather than held, which is what lets null survive being drawn.
	// The tree needs every ticked node present, parents and children alike, so
	// "all" has to be spelled out for it — and spelling it out from the
	// databases in hand means a schema loaded later arrives already ticked,
	// with nothing to write back and no render to chase it.
	const treeSelection = useMemo(
		() => new Set(selected ?? everyKnownId(databases)),
		[selected, databases],
	);

	const handleSelectionChange = (next: Set<string>) => {
		const ids = toDataIds(databases, next);
		// Every database present *by its own id*, not merely as many ids as
		// there are databases: one database sent whole and another sent as a
		// single schema is the same length and is not everything.
		const coversAll =
			databases.length > 0 && databases.every((database) => ids.includes(database.id));
		onChange(coversAll ? null : ids);
	};

	// Named "Filtered" rather than by count, unlike the other two pickers: the
	// count here is of databases and schemas mixed, stored at whichever level
	// describes the selection, so "Selected (2)" could mean two schemas or two
	// whole databases.
	const summary = selected === null ? 'All' : selected.length === 0 ? 'None' : 'Filtered';

	const emptyMessage = failed
		? 'Could not load databases'
		: loading
			? 'Loading databases…'
			: databases.length === 0
				? 'No databases are connected'
				: null;

	return (
		<Popover
			className="w-full"
			align={PopoverAlign.Right}
			// Wider than the other two panels: these are rows of a tree, so a
			// schema is read at an indent and a name truncated at 56 leaves
			// nothing of it.
			panelClassName="w-72 p-1"
			trigger={({ open, toggle: togglePanel }) => (
				<SelectButton
					theme={SelectButtonTheme.SelectField}
					onClick={togglePanel}
					aria-expanded={open}
					aria-label="Filter by database or schema"
				>
					<span className="truncate">{summary}</span>
					<Icon
						name={IconName.ChevronRight}
						className="h-4 w-4 shrink-0 rotate-90 text-secondary dark:text-zinc-400"
					/>
				</SelectButton>
			)}
		>
			{() => (
				<div className="flex flex-col">
					<div className="flex items-center justify-between gap-2 px-2 py-1">
						<span className="text-xs font-medium text-secondary dark:text-zinc-400">
							Filter By
						</span>
						<Button
							theme={ButtonTheme.Minimal}
							size={Size.SMALL}
							type="button"
							onClick={() => onChange(selected === null ? [] : null)}
							disabled={emptyMessage !== null}
						>
							{selected === null ? 'Deselect All' : 'Select All'}
						</Button>
					</div>
					{emptyMessage !== null ? (
						<p className="px-2 py-3 text-center text-xs text-secondary dark:text-zinc-400">
							{emptyMessage}
						</p>
					) : (
						<DataTreeSelect
							databases={databases}
							selectedItems={treeSelection}
							onSelectedItemsChange={handleSelectionChange}
							onLoadSchemas={loadSchemas}
							// Never called: tables are not shown, so nothing
							// asks for them.
							onLoadTables={async () => {}}
							showTables={false}
							framed={false}
						/>
					)}
				</div>
			)}
		</Popover>
	);
};
