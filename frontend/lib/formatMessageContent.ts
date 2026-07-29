// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export enum MessageType {
	TEXT = 'text',
	BOLD = 'bold',
	LINK = 'link',
	LINK_BOLD = 'link_bold',
	CODE = 'code',
	CHART = 'chart',
	CHART_CAROUSEL = 'chart_carousel',
	NEW_LINE = 'new_line',
}

export type ContentSegment =
	| { type: MessageType.TEXT; text: string }
	| { type: MessageType.BOLD; text: string }
	| { type: MessageType.LINK; url: string; text: string }
	| { type: MessageType.LINK_BOLD; url: string; text: string }
	| { type: MessageType.CODE; text: string }
	| { type: MessageType.CHART; text: string }
	| { type: MessageType.CHART_CAROUSEL; text: string }
	| { type: MessageType.NEW_LINE; text: string };

/**
 * Parses a Slack-style mrkdwn string into a flat list of typed segments.
 *
 * Supported syntax:
 *   ```chart ...```         → CHART (ResultChart JSON)
 *   ```chart-carousel ...``` → CHART_CAROUSEL
 *   ```code block```       → CODE
 *   *bold*                 → BOLD
 *   <url|label>            → LINK
 *   *<url|label>*          → LINK_BOLD
 *   "\n"                   → NEW_LINE (between lines, not at the very end)
 *
 * Any remaining text is emitted as TEXT segments.
 */
export const formatAndModifyContent = (input: string): ContentSegment[] => {
	const formattedText = input.trimStart().replace(/\.+$/, '');
	const segments = formattedText.split(/(```[\s\S]*?```)/g);

	return segments.flatMap<ContentSegment>((segment) => {
		if (/^```[\s\S]*```$/.test(segment)) {
			const inner = segment.replace(/^```|```$/g, '');
			const langMatch = inner.match(/^(\S*)\n?([\s\S]*)$/);
			const lang = (langMatch?.[1] ?? '').trim().toLowerCase();
			const body = (langMatch?.[2] ?? '').trimEnd();

			if (lang === 'chart') {
				return [{ type: MessageType.CHART, text: body }];
			}
			if (lang === 'chart-carousel') {
				return [{ type: MessageType.CHART_CAROUSEL, text: body }];
			}

			return [
				{
					type: MessageType.CODE,
					// Preserve prior behaviour: CODE text is the fence body without fences.
					text: inner,
				},
			];
		}

		const lines = segment.split('\n');
		const parts: ContentSegment[] = [];

		lines.forEach((line, index) => {
			let lastIndex = 0;
			const regex = /\*?<([^|]+)\|([^>]+)>\*?|\*([^*]+)\*/g;
			let match: RegExpExecArray | null;

			while ((match = regex.exec(line)) !== null) {
				if (match.index > lastIndex) {
					parts.push({
						type: MessageType.TEXT,
						text: line.slice(lastIndex, match.index),
					});
				}

				const isWrappedInStars = match[0].startsWith('*') && match[0].endsWith('*');

				if (isWrappedInStars && !match[0].includes('<')) {
					parts.push({
						type: MessageType.BOLD,
						text: match[3],
					});
				} else if (match[1] && match[2]) {
					parts.push({
						type: isWrappedInStars ? MessageType.LINK_BOLD : MessageType.LINK,
						url: match[1],
						text: match[2],
					});
				}

				lastIndex = regex.lastIndex;
			}

			if (lastIndex < line.length) {
				parts.push({
					type: MessageType.TEXT,
					text: line.slice(lastIndex),
				});
			}

			if (index < lines.length - 1) {
				parts.push({ type: MessageType.NEW_LINE, text: '' });
			}
		});

		return parts;
	});
};
