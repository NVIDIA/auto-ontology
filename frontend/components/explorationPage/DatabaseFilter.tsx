// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useMemo, useState } from 'react';

import { datasources } from '@/api/datasources';
import { Button, SelectButton } from '@/common/Button';
import { DataTreeSelect } from '@/common/DataTreeSelect';
import { Icon, IconName } from '@/common/icons';
import { Popover } from '@/common/Popover';
import { ButtonTheme, SelectButtonTheme, Size } from '@/enums/button';
import { PopoverAlign } from '@/enums/popover';
import type { Database } from '@/types/datasources';

const DATABASE_SELECTION_NAME = 'exploration-database';

export type DatabaseFilterProps = {
	/** One database id, or null for the unfiltered (all-databases) view. */
	selected: string | null;
	onChange: (selected: string | null) => void;
};

/**
 * Which one database the data-layer Exploration graph may draw.
 *
 * The same tree Discovery's Data filter selects with, stopped at the
 * database and limited to a single choice: the unfiltered catalog is
 * capped, so drawing more than one database at once would silently drop
 * tables. Null is every database (the 500-node view); picking a row loads
 * that catalog in full.
 */
export const DatabaseFilter = ({ selected, onChange }: DatabaseFilterProps) => {
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

	const treeSelection = useMemo(
		() => (selected == null ? new Set<string>() : new Set([selected])),
		[selected],
	);

	const selectedName = databases.find((database) => database.id === selected)?.name;
	const summary = selected === null ? 'All' : (selectedName ?? 'Filtered');

	const emptyMessage = failed
		? 'Could not load databases'
		: loading
			? 'Loading databases…'
			: databases.length === 0
				? 'No databases are connected'
				: null;

	return (
		<Popover
			className="w-40"
			align={PopoverAlign.Right}
			panelClassName="w-72 p-1"
			trigger={({ open, toggle: togglePanel }) => (
				<div className="rounded-lg shadow-md">
					<SelectButton
						theme={SelectButtonTheme.SelectField}
						onClick={togglePanel}
						aria-expanded={open}
						aria-label="Filter by database"
					>
						<span className="flex min-w-0 items-center gap-2">
							<Icon
								name={IconName.Database}
								className="h-4 w-4 shrink-0 text-secondary dark:text-zinc-400"
							/>
							<span className="truncate">{summary}</span>
							{selected !== null && (
								<span
									className="h-1.5 w-1.5 shrink-0 rounded-full bg-red-500"
									aria-hidden
								/>
							)}
						</span>
						<Icon
							name={IconName.ChevronRight}
							className="h-4 w-4 shrink-0 rotate-90 text-secondary dark:text-zinc-400"
						/>
					</SelectButton>
				</div>
			)}
		>
			{({ close }) => (
				<div className="flex flex-col">
					<div className="flex items-center justify-between gap-2 px-2 py-1">
						<span className="text-xs font-medium text-secondary dark:text-zinc-400">
							Filter By
						</span>
						<Button
							theme={ButtonTheme.Minimal}
							size={Size.SMALL}
							type="button"
							onClick={() => {
								onChange(null);
								close();
							}}
							disabled={emptyMessage !== null || selected === null}
						>
							Select All
						</Button>
					</div>
					{emptyMessage !== null ? (
						<p className="px-2 py-3 text-center text-xs text-secondary dark:text-zinc-400">
							{emptyMessage}
						</p>
					) : (
						<>
							<label className="flex min-h-9 cursor-pointer items-center gap-2 rounded-lg px-2 text-sm text-heading hover:bg-zinc-100/90 dark:text-zinc-200 dark:hover:bg-zinc-800/70">
								<span className="inline-flex w-4 shrink-0" />
								<input
									type="radio"
									name={DATABASE_SELECTION_NAME}
									checked={selected === null}
									onChange={() => {
										onChange(null);
										close();
									}}
									className="h-4 w-4 cursor-pointer border-zinc-300 text-[#76b900] focus:ring-[#76b900]/40 dark:border-zinc-600"
								/>
								<span className="inline-flex shrink-0" aria-hidden>
									<Icon
										name={IconName.Database}
										className="h-4 w-4 text-zinc-500 dark:text-zinc-400"
									/>
								</span>
								All databases
							</label>
							<DataTreeSelect
								databases={databases}
								selectedItems={treeSelection}
								onSelectedItemsChange={(next) => {
									const id = databases.find((database) =>
										next.has(database.id),
									)?.id;
									if (id == null) return;
									onChange(id);
									close();
								}}
								onLoadSchemas={async () => []}
								onLoadTables={async () => {}}
								showSchemas={false}
								showTables={false}
								singleSelect
								selectionName={DATABASE_SELECTION_NAME}
								framed={false}
							/>
						</>
					)}
				</div>
			)}
		</Popover>
	);
};
