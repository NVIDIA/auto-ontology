// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Fragment, useMemo } from 'react';
import {
	MessageType,
	formatAndModifyContent,
	type ContentSegment,
} from '@/lib/formatMessageContent';
import { ChartBlock, fenceBareSpecs } from '@/components/ResultChart';

type FormattedContentProps = {
	content: string;
	className?: string;
};

const renderCodeFallback = (text: string, key: number) => (
	<pre
		key={key}
		className="my-2 overflow-x-auto rounded-lg bg-zinc-900 p-3 text-xs leading-relaxed text-[#76b900] dark:bg-zinc-950"
	>
		<code>{text}</code>
	</pre>
);

const renderSegment = (segment: ContentSegment, key: number) => {
	switch (segment.type) {
		case MessageType.TEXT:
			return <Fragment key={key}>{segment.text}</Fragment>;

		case MessageType.BOLD:
			return (
				<strong key={key} className="font-semibold">
					{segment.text}
				</strong>
			);

		case MessageType.LINK:
			return (
				<a
					key={key}
					href={segment.url}
					target="_blank"
					rel="noopener noreferrer"
					className="text-[#76b900] underline hover:text-[#6aa500]"
				>
					{segment.text}
				</a>
			);

		case MessageType.LINK_BOLD:
			return (
				<a
					key={key}
					href={segment.url}
					target="_blank"
					rel="noopener noreferrer"
					className="font-semibold text-[#76b900] underline hover:text-[#6aa500]"
				>
					{segment.text}
				</a>
			);

		case MessageType.CHART:
		case MessageType.CHART_CAROUSEL:
			return (
				<div key={key} className="my-3">
					<ChartBlock
						raw={segment.text}
						fallback={
							<pre className="my-2 overflow-x-auto rounded-lg bg-zinc-900 p-3 text-xs leading-relaxed text-[#76b900] dark:bg-zinc-950">
								<code>{segment.text}</code>
							</pre>
						}
					/>
				</div>
			);

		case MessageType.CODE:
			return renderCodeFallback(segment.text, key);

		case MessageType.NEW_LINE:
			return <br key={key} />;
	}
};

export const FormattedContent = ({ content, className }: FormattedContentProps) => {
	const segments = useMemo(() => formatAndModifyContent(fenceBareSpecs(content)), [content]);

	return (
		<div className={className}>
			{segments.map((segment, index) => renderSegment(segment, index))}
		</div>
	);
};
