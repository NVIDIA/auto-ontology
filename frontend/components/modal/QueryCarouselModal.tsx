// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';

import { Icon, IconName } from '@/components/icons';
import type { ExplorationForeignKey, ExplorationLink } from '@/types/exploration';
import { LabelList } from '@/common/SinglePageComposer';
import { SqlBlock } from '@/components/SqlBlock';
import { Modal } from './Modal';

type QueryCarouselModalProps = {
	link: ExplorationLink | null;
	sourceName: string;
	targetName: string;
	onClose: () => void;
};

const ForeignKeyColumnsPanel = ({
	foreignKeys,
	sourceName,
	targetName,
}: {
	foreignKeys: ExplorationForeignKey[];
	sourceName: string;
	targetName: string;
}) => {
	if (foreignKeys.length === 0) {
		return (
			<p className="text-sm text-zinc-500 dark:text-zinc-400">
				These tables share a foreign key relationship; no stored SQL query references both
				of them together.
			</p>
		);
	}

	return (
		<div className="space-y-4">
			<p className="text-sm text-zinc-500 dark:text-zinc-400">
				No stored SQL query references both tables together. They are joined by the
				following foreign key column{foreignKeys.length > 1 ? 's' : ''}:
			</p>
			{foreignKeys.map((fk, index) => (
				<div
					key={`${fk.sourceColumn}-${fk.targetColumn}-${index}`}
					className="grid grid-cols-1 gap-3 rounded-lg border border-zinc-200 p-3 sm:grid-cols-2 dark:border-zinc-700"
				>
					<div className="min-w-0">
						<p className="truncate text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
							{sourceName}.{fk.sourceColumn}
						</p>
						<div className="mt-1.5">
							<LabelList values={fk.sourceSampleValues ?? []} />
						</div>
					</div>
					<div className="min-w-0">
						<p className="truncate text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
							{targetName}.{fk.targetColumn}
						</p>
						<div className="mt-1.5">
							<LabelList values={fk.targetSampleValues ?? []} />
						</div>
					</div>
				</div>
			))}
		</div>
	);
};

/** Generic modal cycling through the SQL queries (or FK columns) backing a graph edge between two data objects. */
export const QueryCarouselModal = ({
	link,
	sourceName,
	targetName,
	onClose,
}: QueryCarouselModalProps) => {
	const [queryIndex, setQueryIndex] = useState(0);
	const queries = link?.queries ?? [];
	const query = queries[queryIndex] ?? '';
	const viaForeignKey = link?.viaForeignKey ?? false;

	return (
		<Modal open={link != null} onClose={onClose} className="w-full max-w-3xl">
			<header className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-700">
				<div className="flex min-w-0 items-center gap-2">
					<Icon name={IconName.Terms} className="h-5 w-5 shrink-0 text-[#76b900]" />
					<div className="min-w-0">
						<div className="flex items-center gap-2">
							<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
								Query
							</h2>
							{viaForeignKey && (
								<span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[10px] font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
									Foreign key
								</span>
							)}
						</div>
						<p className="truncate text-xs text-zinc-500 dark:text-zinc-400">
							{sourceName} ↔ {targetName}
						</p>
					</div>
				</div>
				<div className="flex items-center gap-3">
					{queries.length > 1 && (
						<div className="flex items-center gap-2">
							<button
								type="button"
								onClick={() => setQueryIndex((index) => Math.max(0, index - 1))}
								disabled={queryIndex === 0}
								className="cursor-pointer rounded p-1 text-zinc-500 hover:bg-zinc-100 disabled:cursor-default disabled:opacity-30 dark:hover:bg-zinc-700"
								aria-label="Previous query"
							>
								←
							</button>
							<span className="text-xs text-zinc-500">
								{queryIndex + 1}/{queries.length}
							</span>
							<button
								type="button"
								onClick={() =>
									setQueryIndex((index) =>
										Math.min(queries.length - 1, index + 1),
									)
								}
								disabled={queryIndex === queries.length - 1}
								className="cursor-pointer rounded p-1 text-zinc-500 hover:bg-zinc-100 disabled:cursor-default disabled:opacity-30 dark:hover:bg-zinc-700"
								aria-label="Next query"
							>
								→
							</button>
						</div>
					)}
					<button
						type="button"
						onClick={onClose}
						className="cursor-pointer rounded p-1 text-lg text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
						aria-label="Close query"
					>
						×
					</button>
				</div>
			</header>
			<div className="max-h-[70dvh] overflow-y-auto p-5">
				{queries.length > 0 ? (
					<SqlBlock sql={query} label="SQL Query" />
				) : (
					<ForeignKeyColumnsPanel
						foreignKeys={link?.foreignKeys ?? []}
						sourceName={sourceName}
						targetName={targetName}
					/>
				)}
			</div>
		</Modal>
	);
};
