// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useMemo, useState } from 'react';
import type { ChatMessage } from '@/types/chat';
import { parseSqlResponse, type ParsedTable } from '@/lib/parseSqlResponse';
import { DynamicTable } from './DynamicTable';
import { FormattedContent } from './FormattedContent';

type SingleCellResult = { column: string; value: string };

const CopyButton = ({ text }: { text: string }) => {
	const [copied, setCopied] = useState(false);

	const handleCopy = useCallback(() => {
		navigator.clipboard.writeText(text).then(() => {
			setCopied(true);
			setTimeout(() => setCopied(false), 2000);
		});
	}, [text]);

	return (
		<button
			type="button"
			onClick={handleCopy}
			className="absolute top-2 right-2 rounded bg-zinc-700 px-2 py-1 text-xs text-zinc-300 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-zinc-600"
		>
			{copied ? 'Copied!' : 'Copy'}
		</button>
	);
};

type QueryResultsSectionProps = {
	sqlResponse: string | undefined;
	parsedTable: ParsedTable | null;
	singleCell: SingleCellResult | null;
};

const QueryResultsSection = ({
	sqlResponse,
	parsedTable,
	singleCell,
}: QueryResultsSectionProps) => (
	<div className="mt-3">
		<div className="mb-1 text-xs font-medium text-zinc-500 dark:text-zinc-400">
			Query results
		</div>
		{!sqlResponse && <p className="text-xs italic text-zinc-400 dark:text-zinc-500">null</p>}
		{sqlResponse && singleCell && (
			<div className="rounded-lg border border-zinc-200 bg-white px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900">
				<span className="text-xs text-zinc-500 dark:text-zinc-400">
					{singleCell.column}:{' '}
				</span>
				<span className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
					{singleCell.value}
				</span>
			</div>
		)}
		{sqlResponse && parsedTable && !singleCell && <DynamicTable table={parsedTable} />}
		{sqlResponse && !parsedTable && (
			<div className="group relative overflow-hidden rounded-lg bg-zinc-900 dark:bg-zinc-950">
				<div className="flex items-center justify-between border-b border-zinc-700 px-3 py-1.5">
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

export const MessageBubble = ({ message }: MessageBubbleProps) => {
	const isUser = message.role === 'user';
	const parsedTable = useMemo(() => parseSqlResponse(message.sqlResponse), [message.sqlResponse]);

	const singleCellResult = useMemo<SingleCellResult | null>(() => {
		if (!parsedTable) return null;
		if (parsedTable.rows.length !== 1 || parsedTable.columns.length !== 1) return null;
		const [column] = parsedTable.columns;
		return { column, value: parsedTable.rows[0][column] ?? '' };
	}, [parsedTable]);

	const showQueryResults = !isUser && (Boolean(message.sql) || Boolean(message.sqlResponse));

	return (
		<div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
			<div
				className={`max-w-[80%] rounded-2xl px-4 py-3 ${
					isUser
						? 'bg-[#76b900] text-white'
						: 'bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100'
				}`}
			>
				{isUser ? (
					<p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
				) : (
					<FormattedContent
						content={message.content}
						className="text-sm leading-relaxed"
					/>
				)}

				{message.sql && (
					<div className="group relative mt-3 overflow-hidden rounded-lg bg-zinc-900 dark:bg-zinc-950">
						<div className="flex items-center justify-between border-b border-zinc-700 px-3 py-1.5">
							<span className="text-xs font-medium text-zinc-400">SQL</span>
							<CopyButton text={message.sql} />
						</div>
						<pre className="overflow-x-auto p-3 text-xs leading-relaxed text-[#76b900]">
							<code>{message.sql}</code>
						</pre>
					</div>
				)}

				{showQueryResults && (
					<QueryResultsSection
						sqlResponse={message.sqlResponse}
						parsedTable={parsedTable}
						singleCell={singleCellResult}
					/>
				)}

				<time className="mt-1.5 block text-right text-[10px] opacity-50">
					{new Date(message.timestamp).toLocaleTimeString([], {
						hour: '2-digit',
						minute: '2-digit',
					})}
				</time>
			</div>
		</div>
	);
};
