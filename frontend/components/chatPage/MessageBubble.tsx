// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useMemo } from 'react';
import type { ChatMessage } from '@/types/chat';
import { CopyButton } from '@/common/Button';
import { formatDate } from '@/common/date';
import { parseSqlResponse, type ParsedTable } from '@/lib/parseSqlResponse';
import { SqlBlock } from '@/common/SqlBlock';
import { DynamicTable } from './DynamicTable';
import { FormattedContent } from './FormattedContent';

type SingleCellResult = { column: string; value: string };

type QueryResultsSectionProps = {
	sqlResponse: string | undefined;
	parsedTable: ParsedTable | null;
	singleCell: SingleCellResult | null;
	className?: string;
};

const QueryResultsSection = ({
	sqlResponse,
	parsedTable,
	singleCell,
	className,
}: QueryResultsSectionProps) => (
	<div className={className}>
		<div className="mb-1 text-xs font-medium text-secondary dark:text-zinc-400">
			Query results
		</div>
		{!sqlResponse && <p className="text-xs italic text-secondary dark:text-zinc-500">null</p>}
		{sqlResponse && singleCell && (
			<div className="rounded-lg border border-zinc-200 bg-white px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900">
				<span className="text-xs text-secondary dark:text-zinc-400">
					{singleCell.column}:{' '}
				</span>
				<span className="text-sm font-semibold text-heading dark:text-zinc-100">
					{singleCell.value}
				</span>
			</div>
		)}
		{sqlResponse && parsedTable && !singleCell && <DynamicTable table={parsedTable} />}
		{sqlResponse && !parsedTable && (
			<div className="group relative overflow-hidden rounded-lg bg-zinc-900 dark:bg-zinc-950">
				<div className="flex items-center justify-between border-b border-zinc-700 px-3 py-1.5">
					{/* Dark in both themes, so the light-theme text tokens do not apply here. */}
					<span className="text-xs font-medium text-zinc-400">Raw</span>
					<CopyButton text={sqlResponse} />
				</div>
				<pre className="overflow-x-auto p-3 text-xs leading-relaxed text-zinc-100">
					<code>{sqlResponse}</code>
				</pre>
			</div>
		)}
	</div>
);

type MessageBubbleProps = {
	message: ChatMessage;
};

/** True when the assistant message embeds a ResultChart fence (illumex Message 2). */
const hasEmbeddedChart = (content: string): boolean =>
	/(^|\n)```(?:chart|chart-carousel)\b/.test(content);

export const MessageBubble = ({ message }: MessageBubbleProps) => {
	const isUser = message.role === 'user';
	const parsedTable = useMemo(() => parseSqlResponse(message.sqlResponse), [message.sqlResponse]);

	const singleCellResult = useMemo<SingleCellResult | null>(() => {
		if (!parsedTable) return null;
		if (parsedTable.rows.length !== 1 || parsedTable.columns.length !== 1) return null;
		const [column] = parsedTable.columns;
		return { column, value: parsedTable.rows[0][column] ?? '' };
	}, [parsedTable]);

	// Illumex layout:
	// Message 1 — text + SQL (no table)
	// Message 2 — charts only, OR table only
	const showCharts = !isUser && hasEmbeddedChart(message.content);
	const showSql = !isUser && Boolean(message.sql) && !showCharts;
	const showQueryResults = !isUser && !showCharts && Boolean(message.sqlResponse);
	const showText = Boolean(message.content.trim());

	// Stretch assistant bubbles that carry SQL / charts / tables to the same
	// width so Message 2 matches Message 1's SQL block.
	const wideAssistant = !isUser && (showSql || showCharts || showQueryResults);

	return (
		<div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
			<div
				className={`max-w-[80%] rounded-2xl px-4 py-3 ${wideAssistant ? 'w-full' : ''} ${
					isUser
						? 'bg-[#76b900] text-white'
						: 'bg-zinc-100 text-heading dark:bg-zinc-800 dark:text-zinc-100'
				}`}
			>
				{showText && (
					<FormattedContent
						content={message.content}
						className="text-sm leading-relaxed"
					/>
				)}

				{showSql && (
					<SqlBlock
						sql={message.sql!}
						label={
							message.sql!.trim().toUpperCase().startsWith('PREDICT') ? 'PQL' : 'SQL'
						}
						className={showText ? 'mt-3' : undefined}
					/>
				)}

				{showQueryResults && (
					<QueryResultsSection
						sqlResponse={message.sqlResponse}
						parsedTable={parsedTable}
						singleCell={singleCellResult}
						className={showText || showSql ? 'mt-3' : undefined}
					/>
				)}

				<time className="mt-1.5 block text-right text-[10px] opacity-50">
					{formatDate(message.timestamp, 'HH:mm')}
				</time>
			</div>
		</div>
	);
};
