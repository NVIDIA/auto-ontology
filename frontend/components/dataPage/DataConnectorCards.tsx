// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useMemo, useState, type ReactNode } from 'react';
import Link from 'next/link';
import { Icon, IconName } from '@/common/icons';
import { OwnerLabel } from '@/common/OwnerLabel';
import { InputsSearch } from '@/common/InputsSearch';
import { formatDate } from '@/common/date';
import { ConnectionType } from '@/enums/connection';
import { ConnectorView } from '@/enums/connectorList';
import { catalogPathFromFocusId } from '@/lib/data/data-catalog-path';
import { connectorsToCsv, downloadTextFile, searchConnectors } from '@/lib/data/connector-list';
import { findOwner } from '@/lib/owners';
import type { Database } from '@/types/datasources';
import type { Owner } from '@/types/owners';

export type DataConnectorCardsProps = {
	databases: Database[];
	owners: Owner[];
};

const CARD_DATE_FORMAT = 'DD MMM, YYYY';

const dateOrDash = (value: string | null | undefined): string =>
	value ? formatDate(value, CARD_DATE_FORMAT) : '—';

const CardRow = ({ label, children }: { label: string; children: ReactNode }) => (
	<div className="flex h-8 items-center justify-between py-0.5 text-sm">
		<span className="min-w-[70px] shrink-0 whitespace-nowrap text-[#5a5e64] dark:text-[#b8b8c1]">
			{label}
		</span>
		<span className="min-w-0 text-[#2b3036] dark:text-[#f0f0f3]">{children}</span>
	</div>
);

/** The icon for a connector's type; the generic database icon when it has none. */
const connectorIcon = (database: Database): IconName =>
	database.dialect === ConnectionType.SNOWFLAKE ? IconName.Snowflake : IconName.Database;

/** One connector: its icon and name, then owner, created and last-pulled. */
const ConnectorCard = ({ database, owners }: { database: Database; owners: Owner[] }) => {
	return (
		<Link
			href={catalogPathFromFocusId(database.id)}
			className="flex flex-col gap-3 rounded-lg border border-[#e9eaea] bg-white pt-5 transition-shadow [&:hover]:shadow-menu focus-visible:shadow-menu focus-visible:outline-none dark:border-[#2e2e33] dark:bg-[#18181b]"
			data-testid={`connector-card-${database.name}`}
		>
			<div className="flex items-center pr-3 pl-4">
				<div className="flex min-w-0 items-center gap-1">
					<Icon
						name={connectorIcon(database)}
						className="h-6 w-6 shrink-0 text-[#171b20] dark:text-white"
					/>
					<span className="truncate text-sm text-[#171b20] dark:text-white">
						{database.name}
					</span>
				</div>
			</div>
			<div className="flex flex-col px-5 pb-5">
				<CardRow label="Owner">
					<OwnerLabel
						owner={findOwner(owners, database.owner_id)}
						brandUnassigned
						inheritColor
					/>
				</CardRow>
				<CardRow label="Created">{dateOrDash(database.created)}</CardRow>
				<CardRow label="Last Pulled">{dateOrDash(database.pulled)}</CardRow>
			</div>
		</Link>
	);
};

/** "System Button Wrap" from the design system: a 36px bordered icon button. */
const ToolbarIconButton = ({
	icon,
	label,
	onClick,
}: {
	icon: IconName;
	label: string;
	onClick?: (event: React.MouseEvent) => void;
}) => (
	<button
		type="button"
		aria-label={label}
		title={label}
		onClick={onClick}
		className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[#e9eaea] bg-white text-[#2b3036] transition-colors hover:bg-[#f1f8e5] hover:text-[#76b900] disabled:text-[#8d9096] dark:border-[#2e2e33] dark:bg-[#18181b] dark:text-[#f0f0f3] dark:hover:bg-[#8ad21a]/12 dark:hover:text-[#8ad21a] dark:disabled:text-[#8a8a94]"
	>
		<Icon name={icon} className="h-6 w-6" />
	</button>
);

/** The list layout: one row per connector, same fields as the card. */
const ConnectorList = ({ databases, owners }: { databases: Database[]; owners: Owner[] }) => {
	// "AO-IL Table Headers": 32px, 20px side padding, semibold, a 1px divider on the right.
	const headerCell =
		'h-8 border-r border-[#e9eaea] bg-[#f4f4f5] px-5 py-1 text-left font-[590] whitespace-nowrap text-[#171b20] last:border-r-0 dark:border-[#2e2e33] dark:bg-[#1b1b1d] dark:text-white';
	// "AO-IL Table Cells": 36px, 20px side padding, a 1px divider underneath.
	const cell = 'h-9 px-5 py-1.5';
	return (
		<div className="overflow-x-auto rounded-lg border border-[#e9eaea] bg-white dark:border-[#2e2e33] dark:bg-[#18181b]">
			<table className="w-full border-collapse text-sm">
				<thead>
					<tr>
						<th className={`${headerCell} w-[88px] text-center`}>Type</th>
						<th className={headerCell}>Name</th>
						<th className={headerCell}>Owner</th>
						<th className={headerCell}>Created At</th>
						<th className={headerCell}>Last Pulled</th>
					</tr>
				</thead>
				<tbody>
					{databases.map((database) => (
						<tr
							key={database.id}
							className="border-b border-[#e9eaea] text-[#2b3036] last:border-b-0 hover:bg-[#f1f8e5] hover:text-[#76b900] dark:border-[#2e2e33] dark:text-[#f0f0f3] dark:hover:bg-[#8ad21a]/12 dark:hover:text-[#8ad21a]"
						>
							<td className={cell}>
								<Icon
									name={connectorIcon(database)}
									className="mx-auto h-6 w-6"
									aria-hidden
								/>
							</td>
							<td className={cell}>
								<Link
									href={catalogPathFromFocusId(database.id)}
									className="block truncate"
								>
									{database.name}
								</Link>
							</td>
							<td className={cell}>
								<OwnerLabel
									owner={findOwner(owners, database.owner_id)}
									brandUnassigned
									inheritColor
								/>
							</td>
							<td className={`${cell} whitespace-nowrap`}>
								{dateOrDash(database.created)}
							</td>
							<td className={`${cell} whitespace-nowrap`}>
								{dateOrDash(database.pulled)}
							</td>
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
};

/** The "All Data" landing page: connectors as cards or a list, with a toolbar. */
export const DataConnectorCards = ({ databases, owners }: DataConnectorCardsProps) => {
	const [query, setQuery] = useState('');
	const [view, setView] = useState<ConnectorView>(ConnectorView.Card);

	const visible = useMemo(() => searchConnectors(databases, query), [databases, query]);

	const handleDownload = () =>
		downloadTextFile(
			'data-connectors.csv',
			connectorsToCsv(visible, owners),
			'text/csv;charset=utf-8',
		);

	return (
		<div className="flex min-h-full flex-col gap-5 bg-[#f5f7f9] px-10 pt-8 pb-10 dark:bg-[#121214]">
			<div className="flex flex-wrap items-start justify-between gap-3">
				<InputsSearch
					value={query}
					onChange={setQuery}
					aria-label="Search data connectors"
					className="w-[382px] max-w-full"
				/>
				<div className="flex items-center gap-2">
					<ToolbarIconButton
						icon={view === ConnectorView.Card ? IconName.ViewCard : IconName.ViewList}
						label={
							view === ConnectorView.Card
								? 'Switch to list view'
								: 'Switch to card view'
						}
						onClick={() =>
							setView(
								view === ConnectorView.Card
									? ConnectorView.List
									: ConnectorView.Card,
							)
						}
					/>
					{/* "Green Outline" from the design system's Buttons page. */}
					<button
						type="button"
						disabled={visible.length === 0}
						onClick={handleDownload}
						className="inline-flex h-9 items-center gap-1 rounded-lg border border-[#76b900] bg-transparent pr-4 pl-2 text-sm text-[#76b900] transition-colors hover:border-transparent hover:bg-[#4f8a00] hover:text-white disabled:cursor-default disabled:border-[#e9eaea] disabled:bg-transparent disabled:text-[#8d9096] dark:border-[#8ad21a] dark:text-[#8ad21a] dark:hover:border-transparent dark:hover:bg-[#8ad21a] dark:hover:text-[#18181b] dark:disabled:border-[#2e2e33] dark:disabled:bg-transparent dark:disabled:text-[#8a8a94]"
					>
						<Icon name={IconName.Excel} className="h-6 w-6" />
						Download
					</button>
				</div>
			</div>
			{visible.length === 0 ? (
				<p className="text-sm text-[#5a5e64] dark:text-[#b8b8c1]">No connectors match.</p>
			) : view === ConnectorView.List ? (
				<ConnectorList databases={visible} owners={owners} />
			) : (
				<div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
					{visible.map((database) => (
						<ConnectorCard key={database.id} database={database} owners={owners} />
					))}
				</div>
			)}
		</div>
	);
};
